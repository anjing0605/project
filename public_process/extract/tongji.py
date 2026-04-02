import os
import csv
from torch_geometric.datasets import Amazon, Planetoid, Coauthor

ROOT = "scratch/project/public_datasets"

DATASETS = [
    ("Amazon", "Computers"),
    ("Amazon", "Photo"),
    ("Coauthor", "CS"),
    ("Coauthor", "Physics"),
    ("Planetoid", "Cora"),
    ("Planetoid", "CiteSeer"),
    ("Planetoid", "PubMed"),
]


def load_dataset(dataset_type, dataset_name):

    if dataset_type == "Amazon":
        dataset = Amazon(
            root=os.path.join(ROOT, "Amazon"),
            name=dataset_name
        )

    elif dataset_type == "Planetoid":
        dataset = Planetoid(
            root=os.path.join(ROOT, "Planetoid"),
            name=dataset_name
        )

    elif dataset_type == "Coauthor":
        dataset = Coauthor(
            root=os.path.join(ROOT, "Coauthor"),
            name=dataset_name
        )

    else:
        raise ValueError("Unsupported dataset")

    return dataset[0]


save_path = "scratch/project/public_process/extract/dataset_statistics.csv"

with open(save_path, "w", newline="") as f:

    writer = csv.writer(f)

    # 写入表头
    writer.writerow([
        "Dataset",
        "Nodes",
        "Edges",
        "Features",
        "Average Degree"
    ])

    for dataset_type, dataset_name in DATASETS:

        data = load_dataset(dataset_type, dataset_name)

        num_nodes = data.num_nodes
        num_edges = data.num_edges // 2
        num_features = data.num_features

        avg_degree = (2 * num_edges) / num_nodes

        writer.writerow([
            dataset_name,
            num_nodes,
            num_edges,
            num_features,
            round(avg_degree, 2)
        ])

print("Saved to:", save_path)