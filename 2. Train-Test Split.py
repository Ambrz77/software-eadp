print("\n[SECTION 2] Train-Test Split...")

combined_data['version'] = combined_data['version'].astype(str)

combined_data = combined_data[combined_data['loc'] > 0]

dataset_pairs = [
    ('ant', '1.3', '1.4'), ('ant', '1.4', '1.5'), ('ant', '1.5', '1.6'),
    ('ant', '1.6', '1.7'), ('camel', '1', '1.2'), ('camel', '1.2', '1.4'),
    ('camel', '1.4', '1.6'), ('ivy', '1.1', '1.4'), ('ivy', '1.4', '2'),
    ('jEdit', '3.2.1', '4'), ('jEdit', '4', '4.1'), ('jEdit', '4.1', '4.2'),
    ('jEdit', '4.2', '4.3'), ('log4j', '1', '1.1'), ('log4j', '1.1', '1.2'),
    ('lucene', '2', '2.2'), ('lucene', '2.2', '2.4'), ('poi', '1.5', '2.0'),
    ('poi', '2.0', '2.5.1'), ('poi', '2.5.1', '3'), ('synapse', '1', '1.1'),
    ('synapse', '1.1', '1.2'), ('velocity', '1.4', '1.5'), ('velocity', '1.5', '1.6.1'),
    ('xalan', '2.4.0', '2.5.0'), ('xalan', '2.5.0', '2.6.0'), ('xalan', '2.6.0', '2.7.0'),
    ('xerces', 'init', '1.2.0'), ('xerces', '1.2.0', '1.3.0'), ('xerces', '1.3.0', '1.4.4')
]


for train_project, train_version, test_version in dataset_pairs:
    print(f"\n[DEBUG] Processing: Train - {train_project} {train_version}, Test - {train_project} {test_version}")

    train_version = normalize_version(train_version)
    test_version  = normalize_version(test_version)

    train_data = combined_data[
        (combined_data['name'] == train_project) &
        (combined_data['version'] == train_version)
    ]

    test_data = combined_data[
        (combined_data['name'] == train_project) &
        (combined_data['version'] == test_version)
    ]

    if train_data.empty:
        print(f"  [WARNING] Train dataset missing for: {train_project} {train_version}")
    if test_data.empty:
        print(f"  [WARNING] Test dataset missing for: {train_project} {test_version}")

    if train_data.empty or test_data.empty:
        print("  [SKIP] This pair is incomplete. Skipping...")
        continue
    
    drop_columns = ['name', 'version', 'bug']
    drop_columns = [col for col in drop_columns if col in train_data.columns]

    X_train = train_data.drop(columns=drop_columns)
    y_train = train_data['bug']
    loc_train = train_data['loc']

    X_test = test_data.drop(columns=drop_columns)
    y_test  = test_data['bug']
    loc_test = test_data['loc']

    print(f"  [DEBUG] X_train shape={X_train.shape}, y_train sum={y_train.sum()}")
    print(f"  [DEBUG] X_test shape={X_test.shape}, y_test sum={y_test.sum()}")
