import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import re

dataset_files = [
    'ant-1.3.csv', 'ant-1.4.csv', 'ant-1.5.csv', 'ant-1.6.csv', 'ant-1.7.csv',
    'camel-1.0.csv', 'camel-1.2.csv', 'camel-1.4.csv', 'camel-1.6.csv',
    'ivy-1.1.csv', 'ivy-1.4.csv', 'ivy-2.0.csv',
    'jedit-3.2.csv', 'jedit-4.0.csv', 'jedit-4.1.csv', 'jedit-4.2.csv', 'jedit-4.3.csv',
    'log4j-1.0.csv', 'log4j-1.1.csv', 'log4j-1.2.csv',
    'lucene-2.0.csv', 'lucene-2.2.csv', 'lucene-2.4.csv',
    'poi-1.5.csv', 'poi-2.0.csv', 'poi-2.5.csv', 'poi-3.0.csv',
    'synapse-1.0.csv', 'synapse-1.1.csv', 'synapse-1.2.csv',
    'velocity-1.4.csv', 'velocity-1.5.csv', 'velocity-1.6.csv',
    'xalan-2.4.csv', 'xalan-2.5.csv', 'xalan-2.6.csv', 'xalan-2.7.csv',
    'xerces-init.csv', 'xerces-1.2.csv', 'xerces-1.3.csv', 'xerces-1.4.csv'
]

data_dir = '/content'

def normalize_version(version):
    match = re.match(r"(\d+(\.\d+)*)", str(version))
    return match.group(0) if match else version

all_datasets = []
skipped_datasets = []

print("[SECTION 1] Loading and Preprocessing Datasets...")

for dataset_name in dataset_files:
    file_path = os.path.join(data_dir, dataset_name)
    try:
        dataset = pd.read_csv(file_path)
        if dataset.empty:
            print(f"[DEBUG] Skipping empty dataset: {dataset_name}")
            skipped_datasets.append(dataset_name)
            continue

        max_bug_in_file = dataset['bug'].max()
        print(f"[DEBUG] {dataset_name}: Loaded shape={dataset.shape}, max bug={max_bug_in_file}")

        dataset['version'] = dataset['version'].apply(normalize_version)

        if 'name.1' in dataset.columns:
            dataset.drop(columns=['name.1'], inplace=True)

        dataset.dropna(subset=['bug', 'loc'], inplace=True)

        all_datasets.append(dataset)

    except Exception as e:
        print(f"[ERROR] Loading {dataset_name}: {e}")
        skipped_datasets.append(dataset_name)

combined_data = pd.concat(all_datasets, ignore_index=True)
print(f"[DEBUG] Combined Dataset Shape: {combined_data.shape}")

# features_to_normalize = [...]
# scaler = MinMaxScaler()
# combined_data[features_to_normalize] = scaler.fit_transform(combined_data[features_to_normalize])

if skipped_datasets:
    print(f"[DEBUG] Skipped datasets: {skipped_datasets}")

print("[DEBUG] Final bug distribution stats:")
print("  - Max bug value:", combined_data['bug'].max())
print("  - Mean bug value:", combined_data['bug'].mean())
print("  - Number of Rows:", len(combined_data))
