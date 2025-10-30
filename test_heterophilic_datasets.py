"""
Test script to verify that heterophilic datasets (Chameleon, Squirrel, Actor) load correctly
"""
import torch
from datasets import NodeDataset

def test_dataset(dataset_name):
    print(f"\n{'='*60}")
    print(f"Testing {dataset_name} dataset")
    print('='*60)

    try:
        dataset = NodeDataset(dataset_name, n_seeds=[0])
        dataset.print_statistics()
        print(f"✓ {dataset_name} loaded successfully!")
        return True
    except Exception as e:
        print(f"✗ {dataset_name} failed to load: {str(e)}")
        return False

if __name__ == '__main__':
    datasets_to_test = ['Chameleon', 'Squirrel', 'Actor']

    print("Testing heterophilic datasets...")
    results = {}

    for dataset_name in datasets_to_test:
        results[dataset_name] = test_dataset(dataset_name)

    print(f"\n{'='*60}")
    print("Summary:")
    print('='*60)
    for dataset_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{dataset_name}: {status}")
