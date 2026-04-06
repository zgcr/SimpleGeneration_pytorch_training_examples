CUDA_VISIBLE_DEVICES=0 torchrun  \
    --nproc_per_node=1 \
    --master_addr 127.0.2.1 \
    --master_port 12001 \
    ../../../tools/test_vqgan_model.py \
    --work-dir ./