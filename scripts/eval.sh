#!/bin/bash
# =============================================================================
# SCRIPT ĐÁNH GIÁ - GLAli (Global & Local Vision-Language Alignment)
# Pipeline: Đánh giá khả năng phát hiện OOD (Out-of-Distribution)
#
# Luồng thực thi:
#   eval.sh  -->  eval_ood_detection.py  -->  Load checkpoint từ output/
#                                         -->  test_ood() trong LocProto
#                                         -->  In AUROC, FPR95, ACC
#
# Trước khi chạy eval.sh, bạn phải đã chạy train.sh để có model checkpoint.
# =============================================================================

# --- [1] TRAINER & CONFIG ---
# Phải khớp với cấu hình đã dùng khi training
TRAINER=LocProto
CSC=True
CTP=end
CFG=vit_b16_ep25
NCTX=16

# --- [2] DATA ---
# Phải trỏ đến cùng thư mục data như khi training
# Đổi thành ./datasets để dùng với BTXRD
DATA=./datasets

# --- [3] TEMPERATURE ---
# T: temperature scaling cho OOD scoring (MCM score = max softmax / T)
# T=1 = không scale (standard softmax)
# T>1 = làm mềm distribution, thường dùng để cải thiện OOD detection
T=1

# --- [4] VÒNG LẶP ĐÁNH GIÁ ---
# Có thể thêm nhiều datasets: "skin40 ISIC btxrd" để so sánh performance
for DATASET in btxrd
do
    for SHOTS in 16   # Phải khớp với SHOTS đã dùng khi training
    do
        for SEED in 1   # Phải khớp với SEED đã dùng khi training
        do
            # eval_ood_detection.py khác train.py:
            # - Không train thêm, chỉ load model đã train
            # - Chạy test_ood() để tính AUROC, FPR@95%TPR
            # - in_dataset: dataset được coi là "in-distribution" (đã train)
            # --root: Thư mục gốc dataset
            # --in_dataset: Dataset "trong phân phối" (đã train)
            # --trainer: Phải khớp với lúc train
            # --dataset-config-file: Config dataset
            # --output-dir: Thư mục output (lưu kết quả eval)
            # --model-dir: Thư mục load checkpoint
            # --load-epoch: Load checkpoint tại epoch 200
            # --T: Temperature scaling
            # DATASET.SUBSAMPLE_CLASSES: Phải khớp với lúc train
            # DATASET.NUM_SHOTS: Phải khớp với lúc train
            CUDA_VISIBLE_DEVICES=0 python eval_ood_detection.py \
            --root ${DATA} \
            --in_dataset ${DATASET} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --seed ${SEED} \
            --output-dir output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}_aaaaaa \
            --model-dir output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}_aaaaaa \
            --load-epoch 200 \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --T ${T} \
            DATASET.SUBSAMPLE_CLASSES base \
            DATASET.NUM_SHOTS ${SHOTS}
        done
    done
done