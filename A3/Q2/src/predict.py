import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from load_dataset import (
    COL761NodeDataset,
    COL761LinkDataset,
    load_dataset,
    _load_edge_list,
)
import models  # noqa: F401  (needed for torch.load to resolve classes)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path: str) -> torch.nn.Module:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = torch.load(model_path, weights_only=False, map_location="cpu")
    model.eval()
    return model


def _random_A(dataset):
    return torch.randint(0, dataset.num_classes, (dataset[0].num_nodes,))

def _random_B(dataset):
    return torch.rand(dataset[0].num_nodes)

def _random_C(V, K):
    return torch.rand(V), torch.rand(V, K)


def _can_full_graph(num_nodes, in_channels, device):
    if device.type != "cuda":
        return num_nodes < 2_000_000
    try:
        bytes_needed = num_nodes * in_channels * 4 * 8
        free_mem, _ = torch.cuda.mem_get_info(device)
        return bytes_needed < free_mem * 0.6
    except Exception:
        return num_nodes < 500_000


@torch.no_grad()
def _infer_labeled_nodes(model, x, edge_index, num_nodes, target_nodes, device):
    """
    Run NeighborLoader inference over ONLY target_nodes (labeled nodes),
    then scatter results back into a full [N] score tensor.
    This avoids running over all 2.89M nodes when we only need ~578K scores.
    """
    loader_data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    workers = 4 if device.type == "cuda" else 0

    loader = NeighborLoader(
        loader_data,
        num_neighbors=[20, 10, 5],
        batch_size=4096,
        input_nodes=target_nodes,
        shuffle=False,
        num_workers=workers,
    )

    logits_list = []
    idx_list = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        logits_list.append(out.cpu())
        idx_list.append(batch.n_id[:batch.batch_size].cpu())

    all_logits = torch.cat(logits_list, dim=0)
    all_idx = torch.cat(idx_list, dim=0)

    num_classes = all_logits.size(1)
    full_logits = torch.zeros((num_nodes, num_classes), dtype=torch.float32)
    full_logits[all_idx] = all_logits
    return full_logits



@torch.no_grad()
def predict_A(model, dataset):
    """Returns predicted class index for every node — LongTensor [N]."""
    data = dataset[0]
    device = get_device()
    model = model.to(device)
    model.eval()

    x = F.normalize(data.x, p=2, dim=1)
    num_nodes = data.num_nodes
    in_channels = x.size(1)

    if _can_full_graph(num_nodes, in_channels, device):
        logits = model(x.to(device), data.edge_index.to(device)).cpu()
    else:
        labeled = data.labeled_nodes
        logits = _infer_labeled_nodes(
            model, x, data.edge_index, num_nodes, labeled, device
        )

    return logits.argmax(dim=1).long()



@torch.no_grad()
def predict_B(model, dataset):
    """Returns positive-class probability for every node — FloatTensor [N]."""
    data = dataset[0]
    device = get_device()
    model = model.to(device)
    model.eval()

    x = F.normalize(data.x, p=2, dim=1)
    num_nodes = data.num_nodes
    in_channels = x.size(1)

    if _can_full_graph(num_nodes, in_channels, device):
        logits = model(x.to(device), data.edge_index.to(device)).cpu()
    else:
        # Only run inference over labeled nodes — evaluate.py only indexes
        # into y_score[labeled_nodes[val_mask]], so unlabeled positions
        # can stay 0 without affecting the AUC score.
        labeled = data.labeled_nodes
        logits = _infer_labeled_nodes(
            model, x, data.edge_index, num_nodes, labeled, device
        )

    if logits.shape[1] == 1:
        scores = torch.sigmoid(logits).squeeze(1)
    else:
        scores = torch.softmax(logits, dim=1)[:, 1]

    return scores.float()




@torch.no_grad()
def _get_split_data_C(dataset, test_dir):
    if test_dir is None:
        pos = dataset.valid_pos
        neg = dataset.valid_neg
        split = "valid"
    else:
        pos = _load_edge_list(os.path.join(test_dir, "test_pos.txt"))
        npy = os.path.join(test_dir, "test_neg_hard.npy")
        with open(npy, "rb") as f:
            neg = torch.from_numpy(np.load(f))
        split = "test"
    return pos, neg, split


@torch.no_grad()
def predict_C(model, dataset, test_dir=None):
    """Returns (pos_scores [P], neg_scores [P, K], split_name)."""
    pos, neg, split = _get_split_data_C(dataset, test_dir)
    P, K, _ = neg.shape

    device = get_device()
    model = model.to(device)
    model.eval()

    x = torch.log1p(dataset.x)
    x = F.normalize(x, p=2, dim=1).to(device)
    edge_index = dataset.edge_index.to(device)

    z = model.encode(x, edge_index)

    pos_scores = model.decode(z, pos.to(device)).cpu()

    chunk_rows = max(1, 200_000 // max(K, 1))
    neg_chunks = []
    for start in range(0, P, chunk_rows):
        end = min(P, start + chunk_rows)
        block = neg[start:end].to(device).reshape(-1, 2)
        block_scores = model.decode(z, block).cpu().reshape(end - start, K)
        neg_chunks.append(block_scores)
    neg_scores = torch.cat(neg_chunks, dim=0)

    return pos_scores, neg_scores, split



def predict_and_save(dataset_name, data_dir, model_path, out_dir,
                     test_dir=None, kerberos="student"):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading dataset {dataset_name} ...")
    ds = load_dataset(dataset_name, data_dir)

    if model_path is not None:
        print(f"Loading model from {model_path} ...")
        model = load_model(model_path)
    else:
        print("No --model_path given — using random predictions.")
        model = None

    if dataset_name == "A":
        y_pred = predict_A(model, ds) if model else _random_A(ds)
        out_path = os.path.join(out_dir, f"{kerberos}_predictions_A.pt")
        torch.save({"y_pred": y_pred}, out_path)
        print(f"Saved {out_path}  shape={y_pred.shape}")

    elif dataset_name == "B":
        y_score = predict_B(model, ds) if model else _random_B(ds)
        out_path = os.path.join(out_dir, f"{kerberos}_predictions_B.pt")
        torch.save({"y_score": y_score}, out_path)
        print(f"Saved {out_path}  shape={y_score.shape}")

    elif dataset_name == "C":
        if model:
            pos_scores, neg_scores, split = predict_C(model, ds, test_dir=test_dir)
        else:
            pos = ds.valid_pos if not test_dir else ds.test_pos
            neg = ds.valid_neg if not test_dir else ds.test_neg
            split = "valid" if not test_dir else "test"
            V, K = pos.shape[0], neg.shape[1]
            pos_scores, neg_scores = _random_C(V, K)

        out_path = os.path.join(out_dir, f"{kerberos}_predictions_C.pt")
        torch.save(
            {"pos_scores": pos_scores, "neg_scores": neg_scores, "split": split},
            out_path,
        )
        print(f"Saved {out_path}  split={split}")
        print(f"  pos_scores : {pos_scores.shape}")
        print(f"  neg_scores : {neg_scores.shape}")


# 
# CLI
# 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["A", "B", "C"])
    parser.add_argument("--task", required=True, choices=["node", "link"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--kerberos", required=True)
    parser.add_argument("--test_dir", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    valid = {"node": ("A", "B"), "link": ("C",)}
    if args.dataset not in valid[args.task]:
        parser.error(f"--task {args.task} invalid for --dataset {args.dataset}")

    if not os.path.isabs(args.data_dir):
        parser.error("--data_dir must be an absolute path")

    model_path = None
    if args.model_dir is not None:
        model_path = os.path.join(
            args.model_dir, f"{args.kerberos}_model_{args.dataset}.pt"
        )

    predict_and_save(
        args.dataset, args.data_dir, model_path, args.output_dir,
        test_dir=args.test_dir, kerberos=args.kerberos,
    )


if __name__ == "__main__":
    main()