# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_port=29541 train_ddp.py --netname convnext --dataset isic --exp_name "0224_convnext_isic" 
# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_port=29542 train_ddp.py --netname convnext --dataset ham --exp_name "0224_convnext_ham" 
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_port=29561 train_ddp.py --netname medmamba --dataset isic --exp_name "0224_medmamba_isic_100epoch" &
# sleep 4h
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_port=29561 train_ddp.py --netname medmamba --dataset ham --exp_name "0224_medmamba_ham_100epoch"
# CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_port=29521 train_ddp.py --netname medmamba --dataset ham --exp_name "0224_medmamba_ham_100epoch" 
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_port=29522 train_ddp.py --netname hifuse --dataset isic --exp_name "0224_hifuse_isic_100epoch" 