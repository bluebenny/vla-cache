# V': VLA-Cache on the SDPA path of B0 (same reuse decisions as V, no attention outputs)
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial    \
    --vla_mode Vp \
    --timing True \
