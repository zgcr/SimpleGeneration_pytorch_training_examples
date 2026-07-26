CUDA_VISIBLE_DEVICES=0 torchrun  \
    --nproc_per_node=1 \
    --master_addr 127.0.1.11 \
    --master_port 10001 \
    ../../../tools/train_vqgan_model_multi_node_nas.py \
    --work-dir ./