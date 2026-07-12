"""
collect_metrics.py - Thu thập tất cả metrics từ pipeline GLAli
Chạy sau khi train.sh hoàn tất.

Chức năng:
  1. Đọc Tensorboard logs → trích xuất training loss/accuracy theo epoch
  2. Chạy OOD evaluation → tính FPR95, AUROC, AUPR
  3. Chạy ID evaluation → tính test accuracy trên base classes
  4. Lưu tất cả metrics vào JSON + CSV
  5. Vẽ biểu đồ training curves + OOD scores distribution

Cách dùng:
  python collect_metrics.py --output-dir <đường_dẫn_output_của_train.sh>

Ví dụ:
  python collect_metrics.py \
    --output-dir output/btxrd/LocProto/vit_b16_ep25_16shots/nctx16_cscTrue_ctpend/seed1_aaaaaa
"""

import argparse
import json
import os
import os.path as osp
import sys
import csv
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

# ============================================================
# 1. TENSORBOARD LOG READER
# ============================================================

def read_tensorboard_logs(tb_dir):
    """Đọc Tensorboard event files và trả về dict {tag: [(step, value)]}."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[WARN] tensorboard not installed. Skipping TB log reading.")
        print("       Install with: pip install tensorboard")
        return {}

    if not osp.isdir(tb_dir):
        print(f"[WARN] Tensorboard dir not found: {tb_dir}")
        return {}

    ea = EventAccumulator(tb_dir)
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]

    print(f"[INFO] Read {len(tags)} tags from Tensorboard: {tags}")
    return data


def aggregate_by_epoch(tb_data, num_batches_per_epoch=None):
    """Gom metrics theo epoch (trung bình mỗi epoch)."""
    epoch_data = {}
    for tag, values in tb_data.items():
        if not values:
            continue
        # Group by epoch
        if num_batches_per_epoch and num_batches_per_epoch > 0:
            epoch_groups = defaultdict(list)
            for step, val in values:
                epoch = step // num_batches_per_epoch
                epoch_groups[epoch].append(val)
            epoch_data[tag] = {ep: np.mean(vals) for ep, vals in sorted(epoch_groups.items())}
        else:
            # Fallback: just use raw data
            epoch_data[tag] = {i: val for i, (step, val) in enumerate(values)}
    return epoch_data


# ============================================================
# 2. OOD EVALUATION (reuses existing code)
# ============================================================

def run_ood_evaluation(output_dir, root="./datasets", dataset="btxrd",
                       trainer="LocProto", cfg_file="configs/trainers/LocProto/vit_b16_ep25.yaml",
                       dataset_cfg="configs/datasets/btxrd.yaml",
                       seed=1, shots=16, load_epoch=200, T=1.0):
    """Chạy OOD evaluation và trả về dict metrics."""
    from dassl.utils import setup_logger, set_random_seed, collect_env_info
    from dassl.config import get_cfg_default
    from dassl.engine import build_trainer
    from utils.detection_util import get_measures
    import clip_w_local
    import trainers.locoop
    import trainers.locproto_supc
    import trainers.zsclip_contra
    import datasets.skin40
    import datasets.ISIC
    import datasets.Dermnet
    import datasets.btxrd as btxrd_module

    # --- Build config ---
    from yacs.config import CfgNode as CN
    cfg = get_cfg_default()

    # Extend cfg
    cfg.TRAINER.LOCOOP = CN()
    cfg.TRAINER.LOCOOP.N_CTX = 16
    cfg.TRAINER.LOCOOP.CSC = False
    cfg.TRAINER.LOCOOP.CTX_INIT = ""
    cfg.TRAINER.LOCOOP.PREC = "fp16"
    cfg.TRAINER.LOCOOP.CLASS_TOKEN_POSITION = "end"
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.Adapter = CN()
    cfg.Adapter.Layer_ID = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    cfg.Adapter.Scale = 1.0
    cfg.Adapter.Down_Rate = 256
    cfg.Adapter.Attn = True
    cfg.Adapter.MLP = True
    cfg.Adapter.Visual = False
    cfg.Adapter.Text = False

    cfg.merge_from_file(dataset_cfg)
    cfg.merge_from_file(cfg_file)

    cfg.DATASET.ROOT = root
    cfg.OUTPUT_DIR = output_dir
    cfg.SEED = seed
    cfg.TRAINER.NAME = trainer
    cfg.DATASET.NUM_SHOTS = shots
    cfg.DATASET.SUBSAMPLE_CLASSES = "base"
    cfg.in_dataset = dataset
    cfg.is_bonder = True
    cfg.is_dense = True
    cfg.lambda_value = 0.99
    cfg.topk = 50

    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    _, preprocess = clip_w_local.load(cfg.MODEL.BACKBONE.NAME)

    # --- Build trainer & load model ---
    print("\n" + "=" * 60)
    print("BUILDING TRAINER FOR EVALUATION")
    print("=" * 60)
    eval_trainer = build_trainer(cfg)
    eval_trainer.load_model(output_dir, epoch=load_epoch)

    if cfg.is_bonder:
        proto_path = osp.join(output_dir, 'proto.pth')
        if osp.exists(proto_path):
            eval_trainer.model.text_prototypes = torch.load(proto_path)

    # --- ID Test Accuracy ---
    print("\n" + "=" * 60)
    print("EVALUATING ID ACCURACY (base classes)")
    print("=" * 60)
    id_accuracy = eval_trainer.test()

    # --- OOD Detection ---
    print("\n" + "=" * 60)
    print("EVALUATING OOD DETECTION")
    print("=" * 60)
    id_data_loader = eval_trainer.dm.id_loader
    ood_loader = eval_trainer.dm.ood_loader

    to_np = lambda x: x.data.cpu().numpy()

    in_scores = eval_trainer.test_ood(id_data_loader, T)
    in_score_mcm = in_scores[0]
    out_scores = eval_trainer.test_ood(ood_loader, T)
    out_score_mcm = out_scores[0]

    auroc, aupr, fpr95 = get_measures(-in_score_mcm, -out_score_mcm)

    ood_metrics = {
        "FPR95": float(fpr95),
        "AUROC": float(auroc),
        "AUPR": float(aupr),
        "ID_accuracy": float(id_accuracy),
        "in_score_mean": float(np.mean(in_score_mcm)),
        "in_score_std": float(np.std(in_score_mcm)),
        "out_score_mean": float(np.mean(out_score_mcm)),
        "out_score_std": float(np.std(out_score_mcm)),
        "num_id_samples": len(in_score_mcm),
        "num_ood_samples": len(out_score_mcm),
    }

    print("\n" + "=" * 60)
    print("OOD DETECTION RESULTS (MCM Score)")
    print("=" * 60)
    print(f"  FPR95:         {fpr95 * 100:.2f}%")
    print(f"  AUROC:         {auroc * 100:.2f}%")
    print(f"  AUPR:          {aupr * 100:.2f}%")
    print(f"  ID Accuracy:   {id_accuracy:.2f}%")
    print(f"  ID samples:    {len(in_score_mcm)}")
    print(f"  OOD samples:   {len(out_score_mcm)}")
    print("=" * 60)

    return ood_metrics, in_score_mcm, out_score_mcm


# ============================================================
# 3. PLOTTING
# ============================================================

def plot_training_curves(epoch_data, save_dir):
    """Vẽ biểu đồ training loss và accuracy."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed. Skipping plots.")
        return

    os.makedirs(save_dir, exist_ok=True)

    # --- Plot 1: All losses ---
    loss_tags = [t for t in epoch_data if 'loss' in t.lower()]
    if loss_tags:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for tag in loss_tags:
            epochs = sorted(epoch_data[tag].keys())
            values = [epoch_data[tag][e] for e in epochs]
            label = tag.replace("train/", "")
            ax.plot(epochs, values, label=label, linewidth=1.5)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Losses', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = osp.join(save_dir, 'training_losses.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[SAVED] {path}")

    # --- Plot 2: Accuracy ---
    acc_tags = [t for t in epoch_data if 'acc' in t.lower()]
    if acc_tags:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for tag in acc_tags:
            epochs = sorted(epoch_data[tag].keys())
            values = [epoch_data[tag][e] for e in epochs]
            label = tag.replace("train/", "")
            ax.plot(epochs, values, label=label, linewidth=1.5, color='green')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Training Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = osp.join(save_dir, 'training_accuracy.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[SAVED] {path}")

    # --- Plot 3: Learning rate ---
    lr_tags = [t for t in epoch_data if 'lr' in t.lower()]
    if lr_tags:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        for tag in lr_tags:
            epochs = sorted(epoch_data[tag].keys())
            values = [epoch_data[tag][e] for e in epochs]
            ax.plot(epochs, values, linewidth=1.5, color='orange')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = osp.join(save_dir, 'learning_rate.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[SAVED] {path}")


def plot_ood_scores(in_scores, out_scores, save_dir):
    """Vẽ histogram phân bố OOD scores."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed. Skipping OOD plots.")
        return

    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(-in_scores, bins=50, alpha=0.6, label='ID (base classes)',
            color='#2196F3', density=True, edgecolor='white')
    ax.hist(-out_scores, bins=50, alpha=0.6, label='OOD (new classes)',
            color='#F44336', density=True, edgecolor='white')
    ax.set_xlabel('Max Softmax Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('OOD Score Distribution (MCM)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = osp.join(save_dir, 'ood_score_distribution.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {path}")


# ============================================================
# 4. SAVE RESULTS
# ============================================================

def save_results(ood_metrics, epoch_data, save_dir):
    """Lưu metrics vào JSON và CSV."""
    os.makedirs(save_dir, exist_ok=True)

    # --- Save OOD metrics JSON ---
    json_path = osp.join(save_dir, 'metrics_summary.json')
    with open(json_path, 'w') as f:
        json.dump(ood_metrics, f, indent=2)
    print(f"[SAVED] {json_path}")

    # --- Save training curves CSV ---
    if epoch_data:
        all_tags = list(epoch_data.keys())
        all_epochs = set()
        for tag_data in epoch_data.values():
            all_epochs.update(tag_data.keys())
        all_epochs = sorted(all_epochs)

        csv_path = osp.join(save_dir, 'training_curves.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = ['epoch'] + [t.replace('train/', '') for t in all_tags]
            writer.writerow(headers)
            for ep in all_epochs:
                row = [ep] + [epoch_data[t].get(ep, '') for t in all_tags]
                writer.writerow(row)
        print(f"[SAVED] {csv_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Collect all GLAli metrics")
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output dir from train.sh (contains tensorboard/, attn_learner/, ...)')
    parser.add_argument('--root', type=str, default='./datasets',
                        help='Dataset root directory')
    parser.add_argument('--dataset', type=str, default='btxrd')
    parser.add_argument('--trainer', type=str, default='LocProto')
    parser.add_argument('--cfg-file', type=str,
                        default='configs/trainers/LocProto/vit_b16_ep25.yaml')
    parser.add_argument('--dataset-cfg', type=str,
                        default='configs/datasets/btxrd.yaml')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--shots', type=int, default=16)
    parser.add_argument('--load-epoch', type=int, default=200)
    parser.add_argument('--T', type=float, default=1.0, help='Temperature for MCM')
    parser.add_argument('--skip-ood', action='store_true',
                        help='Skip OOD evaluation (only collect training metrics)')
    parser.add_argument('--num-batches', type=int, default=None,
                        help='Number of batches per epoch (for TB log aggregation)')
    args = parser.parse_args()

    report_dir = osp.join(args.output_dir, 'report')
    os.makedirs(report_dir, exist_ok=True)

    # ---- Step 1: Tensorboard logs ----
    print("\n" + "#" * 60)
    print("# STEP 1: Reading Tensorboard Logs")
    print("#" * 60)
    tb_dir = osp.join(args.output_dir, 'tensorboard')
    tb_data = read_tensorboard_logs(tb_dir)
    epoch_data = aggregate_by_epoch(tb_data, args.num_batches)

    if epoch_data:
        plot_training_curves(epoch_data, report_dir)
    else:
        print("[WARN] No Tensorboard data found. Training plots skipped.")

    # ---- Step 2: OOD Evaluation ----
    ood_metrics = {}
    in_scores, out_scores = None, None

    if not args.skip_ood:
        print("\n" + "#" * 60)
        print("# STEP 2: Running OOD Evaluation")
        print("#" * 60)
        ood_metrics, in_scores, out_scores = run_ood_evaluation(
            output_dir=args.output_dir,
            root=args.root,
            dataset=args.dataset,
            trainer=args.trainer,
            cfg_file=args.cfg_file,
            dataset_cfg=args.dataset_cfg,
            seed=args.seed,
            shots=args.shots,
            load_epoch=args.load_epoch,
            T=args.T
        )
        plot_ood_scores(in_scores, out_scores, report_dir)
    else:
        print("\n[INFO] OOD evaluation skipped (--skip-ood)")

    # ---- Step 3: Save everything ----
    print("\n" + "#" * 60)
    print("# STEP 3: Saving Results")
    print("#" * 60)
    save_results(ood_metrics, epoch_data, report_dir)

    # ---- Final Summary ----
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    if ood_metrics:
        print(f"  ID Accuracy:   {ood_metrics.get('ID_accuracy', 'N/A'):.2f}%")
        print(f"  FPR95:         {ood_metrics.get('FPR95', 0) * 100:.2f}%")
        print(f"  AUROC:         {ood_metrics.get('AUROC', 0) * 100:.2f}%")
        print(f"  AUPR:          {ood_metrics.get('AUPR', 0) * 100:.2f}%")
    print(f"\n  Results saved to: {report_dir}/")
    print(f"    - metrics_summary.json")
    if epoch_data:
        print(f"    - training_curves.csv")
        print(f"    - training_losses.png")
        print(f"    - training_accuracy.png")
        print(f"    - learning_rate.png")
    if in_scores is not None:
        print(f"    - ood_score_distribution.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
