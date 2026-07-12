#!/bin/bash
# =============================================================================
# SCRIPT THU THẬP METRICS - GLAli
# Chạy sau khi train.sh hoàn tất
#
# Sẽ:
#   1. Đọc Tensorboard logs → training loss/accuracy curves
#   2. Load model checkpoint → chạy OOD evaluation
#   3. Tính FPR95, AUROC, AUPR, ID Accuracy
#   4. Lưu JSON + CSV + biểu đồ PNG vào output/.../report/
# =============================================================================

# --- CẤU HÌNH (phải khớp với train.sh) ---
TRAINER=LocProto
CFG=vit_b16_ep25
DATASET=btxrd
DATA=./datasets
SHOTS=16
SEED=1
NCTX=16
CSC=True
CTP=end
LOAD_EPOCH=200
T=1

# --- OUTPUT DIR (khớp với train.sh) ---
DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}_aaaaaa

echo "============================================"
echo "  Collecting metrics from: ${DIR}"
echo "============================================"

python collect_metrics.py \
    --output-dir ${DIR} \
    --root ${DATA} \
    --dataset ${DATASET} \
    --trainer ${TRAINER} \
    --cfg-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --dataset-cfg configs/datasets/${DATASET}.yaml \
    --seed ${SEED} \
    --shots ${SHOTS} \
    --load-epoch ${LOAD_EPOCH} \
    --T ${T}
