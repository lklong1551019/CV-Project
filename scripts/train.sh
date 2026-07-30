#!/bin/bash
# =============================================================================
# SCRIPT HUẤN LUYỆN - GLAli (Global & Local Vision-Language Alignment)
# Pipeline: Few-Shot Learning + Few-Shot OOD Detection trên ảnh y tế
#
# Luồng thực thi:
#   train.sh  -->  train.py  -->  DataManager  -->  BTXRD dataset loader
#                                               -->  CustomCLIP (ViT-B/16)
#                                               -->  LocProto Trainer
#                                               -->  output/ (model checkpoint)
# =============================================================================

# --- [1] TRAINER ---
# LocProto là trainer chính của GLAli, định nghĩa trong trainers/locproto_supc.py
# Gồm 3 thành phần:
#   - Bonder (CrossAttn): tinh chỉnh text embeddings bằng thông tin ảnh cục bộ
#   - Dense alignment: căn chỉnh local image patch với text embeddings
#   - Supervised Contrastive Loss: phân biệt patch in-distribution vs OOD
TRAINER=LocProto

# --- [2] DATA & DATASET ---
# DATA: thư mục gốc chứa tất cả datasets.
# btxrd.py sẽ nối thêm "btxrd/" vào đây.
# Cấu trúc: ./datasets/btxrd/{train,val,test}/{images,annotations}
DATA=./datasets

# DATASET: tên dataset, phải khớp với tên class trong DATASET_REGISTRY
# (đăng ký bằng @DATASET_REGISTRY.register() trong datasets/btxrd.py)
# Tên này cũng xác định file config: configs/datasets/${DATASET}.yaml
DATASET=btxrd

# --- [3] MODEL CONFIG ---
# CFG: file cấu hình mô hình (backbone, optimizer, batch size, epochs...)
# Xem chi tiết: configs/trainers/LocProto/vit_b16_ep25.yaml
# Mặc dù tên là "ep25", file yaml thực tế đặt MAX_EPOCH=200
CFG=vit_b16_ep25

# --- [4] PROMPT LEARNING ---
# CTP: vị trí class token trong chuỗi prompt
#   end  -> [ctx_1]...[ctx_NCTX][class_name]
#   middle -> [ctx_1]...[class_name]...[ctx_NCTX]
CTP=end

# NCTX: số lượng context token học được (learnable prompt tokens)
# Ví dụ NCTX=16: model học 16 token bổ sung quanh tên class trong text prompt
NCTX=16

# CSC: Class-Specific Context
#   True  = mỗi class có bộ prompt riêng (nhiều tham số hơn)
#   False = dùng chung một bộ context token cho tất cả classes
CSC=True

# --- [5] HYPERPARAMETERS CỦA GLAli ---
# lambda (alpha): hệ số trong công thức cập nhật text prototype
#   updated_proto = alpha * original_proto + (1 - alpha) * refined_proto
#   lambda=0.99 -> giữ 99% embedding gốc, chỉ tinh chỉnh 1% từ visual feedback
#   Giá trị chuẩn theo paper: alpha = 0.99
lambda=0.99

# topk: số local patches được chọn để căn chỉnh với text
#   ViT-B/16 tạo ra 196 patches (14x14 grid)
#   topk=50 chọn 50 patches có entropy thấp nhất (patch tự tin nhất về dự đoán)
topk=40

# =============================================================================
# VÒNG LẶP HUẤN LUYỆN
# Hỗ trợ chạy nhiều seeds và nhiều số shots khác nhau
# =============================================================================
for SEED in 1   # Seed ngẫu nhiên. Thêm nhiều giá trị: "1 2 3" để chạy 3 seeds
do
    for SHOTS in 16  # Số ảnh/class dùng để train (few-shot thường: 1/2/4/8/16)
    do
        # Thư mục lưu kết quả: model checkpoint, tensorboard logs, metrics
        DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}_aaaaaa

        echo $PWD   # In ra working directory hiện tại để debug

        # --root: Thư mục gốc dataset
        # --seed: Seed để tái lập kết quả
        # --trainer: Class trainer sử dụng
        # --dataset-config-file: Config dataset (tên class...)
        # --config-file: Config model (LR, epochs...)
        # --output-dir: Nơi lưu model và logs
        # --lambda_value: Hệ số alpha text prototype
        # --topk: Số local patches align
        # --is_bonder: Bật module Bonder (CrossAttn)
        # --is_dense: Bật Dense Local Alignment
        # --is_sc: Bật Supervised Contrastive Loss
        # TRAINER.LOCOOP.N_CTX: Số context tokens
        # TRAINER.LOCOOP.CSC: Class-specific context flag
        # TRAINER.LOCOOP.CLASS_TOKEN_POSITION: Vị trí class token
        # DATASET.NUM_SHOTS: Số shots -> lấy mẫu từ train
        # DATASET.SUBSAMPLE_CLASSES: "base": nửa đầu classes để train, "new":  nửa sau classes (unseen), "all":  toàn bộ classes
        CUDA_VISIBLE_DEVICES=0 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --lambda_value ${lambda} \
        --topk ${topk} \
        --is_bonder True \
        --is_dense True \
        --is_sc True \
        TRAINER.LOCOOP.N_CTX ${NCTX} \
        TRAINER.LOCOOP.CSC ${CSC} \
        TRAINER.LOCOOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES base
    done
done