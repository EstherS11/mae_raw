#!/bin/bash
    # --pretrain_mask_mlp_adapter /data/zfr/llava_modelarts/out/llava-v1.5-7b-pretrain_pub/mask_projector.bin \
    # --pretrain_mm_mlp_adapter /data/zfr/llava_modelarts/out/llava-v1.5-7b-pretrain_pub/mm_projector.bin \
#!/bin/bash
# deepspeed llava/train/train_mem.py \
export MASTER_PORT=34118
GPUS=${GPUS:-4}
export DS_SKIP_CUDA_CHECK=1
export PATH=/usr/local/cuda/bin:$PATH
# deepspeed llava/train/train_mem.py \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    main_finetune.py \
        --world_size 1 \
        --dist_eval \
        --batch_size 32 \
        --epochs 200 \
        --output_dir ../mae_raw_ckpt/output_dir_336_zuobi_v3_dice/ \
        --log_dir ./mae_raw_ckpt/output_dir_336_zuobi_v3_dice/  \
        --num_workers 16 \
        --input_size 336