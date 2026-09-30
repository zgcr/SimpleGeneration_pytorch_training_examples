CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_addr 127.0.1.2 \
    --master_port 10002 \
    ../../../tools/train_universal_generation_edit_model_fsdp_multi_node_nas.py \
    --work-dir ./
