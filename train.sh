# CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29506 train_HAM_spaWithOurs.py --lr 0.00005 --expname '0225_HAM_spaWithOurs_lr5e-5_8head_newdata' 
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29538 train_isic19_resnet50WithOurs.py --lr 0.00005 --expname '0226_ablation_withRADGM_withIBFA' 
