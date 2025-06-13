from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix

def eadp_enhanced_metrics(predictions, y_test, loc_test, total_bugs, total_loc, num_modules):

    idx_sorted = np.argsort(-predictions)
    y_sorted = y_test[idx_sorted]
    loc_sorted = loc_test[idx_sorted]

    cutoff = 0.2 * total_loc
    csum_loc = np.cumsum(loc_sorted)
    top_mask = (csum_loc <= cutoff)

    k = np.count_nonzero(y_sorted[top_mask] > 0)
    n = np.sum(y_sorted[top_mask])
    m = np.sum(top_mask)
    K = np.count_nonzero(y_test>0)
    N = np.sum(y_test)

    recall_20 = (k / K) if K else 0.0
    pofb_20   = (n / N) if N else 0.0
    prec_20   = (k / m) if m else 0.0
    pmi_20    = (m / num_modules) if num_modules else 0.0

    first_bug_idx=None
    for i in range(len(y_sorted)):
        if y_sorted[i]>0:
            first_bug_idx=i
            break
    IFA = np.count_nonzero(y_sorted[:first_bug_idx]==0) if first_bug_idx is not None else m

    idx_opt   = np.argsort(-y_test)
    idx_worst = np.argsort(y_test)

    def area_under_curve(indices):
        cum_loc=0.0; cum_bug=0.0; area=0.0
        for idx in indices:
            prev_loc_frac = cum_loc/total_loc
            prev_bug_frac = cum_bug/N if N else 0.0
            cum_loc+=loc_test[idx]
            cum_bug+=y_test[idx]
            loc_frac = cum_loc/total_loc
            bug_frac = cum_bug/N if N else 0.0
            area+=0.5*(bug_frac+prev_bug_frac)*(loc_frac-prev_loc_frac)
        return area

    area_pred  = area_under_curve(idx_sorted)
    area_opt   = area_under_curve(idx_opt)
    area_worst = area_under_curve(idx_worst)
    popt = (area_pred-area_worst)/(area_opt-area_worst) if (area_opt-area_worst)!=0 else 1.0

    y_true_bin = (y_test>0).astype(int)
    y_pred_bin = (predictions>=0.5).astype(int)

    from math import sqrt
    f1= f1_score(y_true_bin,y_pred_bin) if np.unique(y_true_bin).size>1 else 0.0
    try:
        auc= roc_auc_score(y_true_bin,predictions)
    except ValueError:
        auc=0.0

    cm= confusion_matrix(y_true_bin,y_pred_bin,labels=[0,1])
    if cm.shape==(2,2):
        TN,FP,FN,TP= cm.ravel()
    else:
        TN=FP=FN=TP=0
    sens= TP/(TP+FN) if (TP+FN) else 0.0
    spec= TN/(TN+FP) if (TN+FP) else 0.0
    g_measure= sqrt(sens*spec) if sens and spec else 0.0
    mcc= matthews_corrcoef(y_true_bin,y_pred_bin) if np.unique(y_true_bin).size>1 else 0.0

    return {
        "Recall@20%": recall_20,
        "PofB@20%": pofb_20,
        "Precision@20%": prec_20,
        "PMI@20%": pmi_20,
        "IFA": float(IFA),
        "Popt": popt,
        "F1 Score": f1,
        "G-Measure": g_measure,
        "AUC": auc,
        "MCC": mcc
    }

results_superior=[]

def run_superior_methodology_evaluation(
    model_choice="xgb",
    attention_epochs=5,
    attention_lr=1e-3,
    attention_softmax=True,
    use_contrastive=True,
    contrastive_margin=1.5,
    contrastive_weight=0.2,
    init_alpha=None,
    raw_transform="log",
    enable_class_weight=False,
    class_weight_factor=1.0,
    prediction_offset=0.0,
    enable_synergy_7=False,
    synergy_include=None,
    synergy_factors=None,
    runs=20
):

    global results_superior
    results_superior=[]

    print("[INFO] Running Superior EADP Evaluation with parameters:")
    print(f"       model_choice={model_choice}, raw_transform={raw_transform}, "
          f"attention_epochs={attention_epochs}, attention_lr={attention_lr}, "
          f"softmax={attention_softmax}, use_contrastive={use_contrastive}, "
          f"contrastive_margin={contrastive_margin}, contrastive_weight={contrastive_weight},")
    print(f"       init_alpha={'provided' if init_alpha else 'None'}, synergy_7={enable_synergy_7}, "
          f"enable_class_weight={enable_class_weight}, class_weight_factor={class_weight_factor},")
    print(f"       synergy_include={synergy_include}, synergy_factors={synergy_factors}, "
          f"prediction_offset={prediction_offset}")

    for (train_proj, train_ver, test_ver) in dataset_pairs:
        train_df= combined_data[(combined_data['name']==train_proj)&(combined_data['version']==train_ver)]
        test_df= combined_data[(combined_data['name']==train_proj)&(combined_data['version']==test_ver)]
        if train_df.empty or test_df.empty:
            print(f"  [SKIP] {train_proj}-{train_ver}->{test_ver}")
            continue

        drop_cols= [c for c in ['name','version','bug'] if c in train_df.columns]
        X_train= train_df.drop(columns=drop_cols).to_numpy()
        y_train= train_df['bug'].to_numpy()
        loc_train=train_df['loc'].to_numpy()

        X_test= test_df.drop(columns=drop_cols).to_numpy()
        y_test= test_df['bug'].to_numpy()
        loc_test= test_df['loc'].to_numpy()

        total_bugs_test= np.sum(y_test)
        total_loc_test = np.sum(loc_test)
        num_modules_test= len(y_test)

        run_metrics_list=[]
        for run_i in range(runs):
            seed=42+run_i
            model_obj, pred_func = train_superior_model(
                X_train, y_train,
                model_choice=model_choice,
                attention_epochs=attention_epochs,
                attention_lr=attention_lr,
                attention_softmax=attention_softmax,
                use_contrastive=use_contrastive,
                contrastive_margin=contrastive_margin,
                contrastive_weight=contrastive_weight,
                init_alpha=init_alpha,
                raw_transform=raw_transform,
                enable_synergy_7=enable_synergy_7,
                synergy_include=synergy_include,
                synergy_factors=synergy_factors,
                enable_class_weight=enable_class_weight,
                class_weight_factor=class_weight_factor,
                prediction_offset=prediction_offset,
                random_seed=seed
            )

            probs_test= pred_func(X_test)
            mets= eadp_enhanced_metrics(
                probs_test, y_test, loc_test,
                total_bugs_test, total_loc_test, num_modules_test
            )
            run_metrics_list.append(mets)

        if run_metrics_list:
            final_mets={}
            for key in run_metrics_list[0].keys():
                final_mets[key]= float(np.mean([m[key] for m in run_metrics_list]))
        else:
            final_mets={k:0 for k in["Recall@20%","PofB@20%","Precision@20%",
                                     "PMI@20%","IFA","Popt","F1 Score","G-Measure","AUC","MCC"]}

        pair_str=f"{train_proj}-{train_ver}->{test_ver}"
        results_superior.append({"pair":pair_str,"model":model_choice,**final_mets})
        print(f"  [RESULT] {pair_str} => {final_mets}")

    if not results_superior:
        print("[INFO] No results to summarize.")
        return

    metric_names=["Recall@20%","PofB@20%","Precision@20%","PMI@20%","IFA","Popt","F1 Score","G-Measure","AUC","MCC"]
    summary={mn:float(np.mean([row[mn] for row in results_superior])) for mn in metric_names}

    print("\n[INFO] Overall Average EADP Metrics (Superior Approach):")
    for mn in metric_names:
        print(f"  {mn}: {summary[mn]:.4f}")
    print("[INFO] Completed Superior EADP Evaluation.")


run_superior_methodology_evaluation(
    model_choice="xgb",
    synergy_include=[
        True, True, True, True, False,
        True, False, True, True, True
    ],
    synergy_factors=[
        1.2, 1.1, 1.0, 1.1, 0.8,
        1.2, 1.0, 1.2, 1.1, 1.0
    ],
    runs=20
)

run_superior_methodology_evaluation(
    model_choice="ada",
    synergy_include=[
        True, True, True, True, False,
        True, False, True, True, True
    ],
    synergy_factors=[
        1.2, 1.1, 1.0, 1.1, 0.8,
        1.2, 1.0, 1.2, 1.1, 1.0
    ],
    runs=20
)

run_superior_methodology_evaluation(
    model_choice="deep",
    synergy_include=[
        True, True, True, True, False,
        True, False, True, True, True
    ],
    synergy_factors=[
        1.2, 1.1, 1.0, 1.1, 0.8,
        1.2, 1.0, 1.2, 1.1, 1.0
    ],
    runs=20
)
