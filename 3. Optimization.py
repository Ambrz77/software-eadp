import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def synergy_fusion_combine_transform(
    X_in,
    enable_synergy_7=False,
    raw_transform="none",
    synergy_include=None,
    synergy_factors=None
):
    if hasattr(X_in, "values"):
        X_in = X_in.values

    if raw_transform == "log":
        X_in = np.log1p(X_in)

    if synergy_include is None:
        synergy_include = [True]*10
    if synergy_factors is None:
        synergy_factors = [1.0]*10

    if len(synergy_include) != 10 or len(synergy_factors) != 10:
        raise ValueError("synergy_include and synergy_factors must be length 10.")

    def safe_col(arr, idx, default=0.0):
        if idx < arr.shape[1]:
            return arr[:, idx]
        return np.full(arr.shape[0], default)


    wmc    = safe_col(X_in, 0)
    cbo    = safe_col(X_in, 3)
    rfc    = safe_col(X_in, 4)
    lcom   = safe_col(X_in, 5)
    ce     = safe_col(X_in, 7)
    npm    = safe_col(X_in, 8)
    loc    = safe_col(X_in, 10)
    moa    = safe_col(X_in, 12)
    max_cc = safe_col(X_in, 18)


    synergy_1 = np.log1p(wmc * rfc)
    synergy_2 = np.log1p(loc * max_cc)
    synergy_3 = np.log1p(cbo * rfc)
    synergy_4 = np.log1p(moa * npm)
    synergy_5 = np.log1p(lcom * npm)
    synergy_6 = np.log1p(ce * (loc + 1.0))
    synergy_7 = (rfc + cbo + ce) if enable_synergy_7 else np.zeros_like(rfc)

    ratio_1   = rfc / (wmc + 1.0)
    ratio_2   = ce  / (loc + 1.0)
    ratio_3   = (loc + 1.0) / (wmc + 1.0)

    synergy_values = [
        synergy_1, synergy_2, synergy_3, synergy_4, synergy_5,
        synergy_6, synergy_7, ratio_1, ratio_2, ratio_3
    ]

    for i in range(10):
        synergy_values[i] *= synergy_factors[i]

    synergy_filtered = [
        synergy_values[i] for i in range(10) if synergy_include[i]
    ]

    if len(synergy_filtered) > 0:
        synergy_block = np.column_stack(synergy_filtered)
    else:
        synergy_block = np.zeros((X_in.shape[0], 0))

    X_aug = np.hstack((X_in, synergy_block))
    return X_aug

def attention_train_transform(
    X_train_aug,
    y_train,
    n_epochs=10,
    lr=1e-3,
    apply_softmax=True,
    use_contrastive=True,
    contrastive_margin=1.0,
    contrastive_weight=0.1,
    init_alpha=None,
    random_seed=42
):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    y_bin = (y_train > 0).astype(np.float32)

    X_torch = torch.tensor(X_train_aug, dtype=torch.float32)
    y_torch = torch.tensor(y_bin,       dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_torch, y_torch)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = X_train_aug.shape[1]

    if init_alpha is not None:
        alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32), requires_grad=True)
    else:
        alpha = nn.Parameter(torch.zeros(input_dim), requires_grad=True)

    optimizer = optim.Adam([alpha], lr=lr)

    bce_loss_fn = nn.BCELoss()

    def logistic_prob(xb):
        if apply_softmax:
            alpha_norm = torch.softmax(alpha, dim=0)
            logit = (xb * alpha_norm).sum(dim=1)
        else:
            logit = (xb * alpha).sum(dim=1)
        return torch.sigmoid(logit)

    def contrastive_loss(embeddings, labels):
        bs = embeddings.size(0)
        if bs < 2:
            return torch.tensor(0.0, device=embeddings.device)

        diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)

        eq_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        loss_sim = eq_mask * dist_sq
        loss_dissim = (1 - eq_mask) * torch.clamp(
            contrastive_margin - torch.sqrt(dist_sq + 1e-6), min=0.0
        )**2

        mask = torch.ones_like(loss_sim) - torch.eye(bs, device=embeddings.device)

        return ((loss_sim + loss_dissim) * mask).sum() / (mask.sum() + 1e-6)

    for epoch in range(n_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            probs = logistic_prob(xb)
            loss_bce = bce_loss_fn(probs, yb)

            loss_total = loss_bce
            if use_contrastive:
                if apply_softmax:
                    alpha_norm = torch.softmax(alpha, dim=0)
                    attended = xb * alpha_norm
                else:
                    attended = xb * alpha
                loss_ctr = contrastive_loss(attended, yb)
                loss_total = loss_bce + contrastive_weight * loss_ctr

            loss_total.backward()
            optimizer.step()

    alpha_final = alpha.detach().cpu().numpy().copy()
    if apply_softmax:
        exps = np.exp(alpha_final)
        alpha_final = exps / np.sum(exps)

    def attention_transform(X_in_aug):
        return X_in_aug * alpha_final

    return attention_transform

def train_superior_model(
    X_train,
    y_train,
    model_choice="xgb",
    attention_epochs=5,
    attention_lr=1e-3,
    attention_softmax=True,
    use_contrastive=True,
    contrastive_margin=1.0,
    contrastive_weight=0.1,
    init_alpha=None,
    raw_transform="log",
    enable_synergy_7=False,
    synergy_include=None,
    synergy_factors=None,
    enable_class_weight=False,
    class_weight_factor=1.0,
    prediction_offset=0.0,
    random_seed=42
):
    X_train_aug = synergy_fusion_combine_transform(
        X_train,
        enable_synergy_7=enable_synergy_7,
        raw_transform=raw_transform,
        synergy_include=synergy_include,
        synergy_factors=synergy_factors
    )

    attention_fn = attention_train_transform(
        X_train_aug, y_train,
        n_epochs=attention_epochs,
        lr=attention_lr,
        apply_softmax=attention_softmax,
        use_contrastive=use_contrastive,
        contrastive_margin=contrastive_margin,
        contrastive_weight=contrastive_weight,
        init_alpha=init_alpha,
        random_seed=random_seed
    )

    X_train_att = attention_fn(X_train_aug)

    y_bin = (y_train > 0).astype(int)

    pos_count = np.sum(y_bin)
    neg_count = len(y_bin) - pos_count
    scale_pos_weight = 1.0
    if enable_class_weight and pos_count > 0:
        scale_pos_weight = (neg_count / float(pos_count)) * class_weight_factor

    if model_choice == "xgb":
        param_xgb = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_seed,
            "eval_metric": "logloss"
        }
        if enable_class_weight and pos_count > 0:
            param_xgb["scale_pos_weight"] = scale_pos_weight
        clf = XGBClassifier(**param_xgb)

    elif model_choice == "ada":
        base_params = {"max_depth": 4, "random_state": random_seed}
        if enable_class_weight and pos_count > 0:
            base_params["class_weight"] = {0:1.0, 1:scale_pos_weight}
        base_tree = DecisionTreeClassifier(**base_params)
        clf = AdaBoostClassifier(
            estimator=base_tree,
            n_estimators=200,
            learning_rate=0.1,
            random_state=random_seed
        )

    elif model_choice == "deep":
        cw = None
        if enable_class_weight and pos_count > 0:
            cw = {0:1.0, 1:scale_pos_weight}
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=2,
            class_weight=cw,
            random_state=random_seed
        )

    else:
        raise ValueError("model_choice must be in {'xgb','ada','deep'}.")

    clf.fit(X_train_att, y_bin)

    def predict_function(X_new):
        X_new_aug = synergy_fusion_combine_transform(
            X_new,
            enable_synergy_7=enable_synergy_7,
            raw_transform=raw_transform,
            synergy_include=synergy_include,
            synergy_factors=synergy_factors
        )
        X_new_att = attention_fn(X_new_aug)
        probs = clf.predict_proba(X_new_att)[:,1]

        if prediction_offset != 0.0:
            return np.clip(probs + prediction_offset, 0.0, 1.0)
        return probs

    return clf, predict_function
