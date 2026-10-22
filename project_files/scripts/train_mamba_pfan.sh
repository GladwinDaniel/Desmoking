set -ex
python train.py --dataroot  ./datasets/composite  \
    --name mamba_pfan    \
    --model pix2pix \
    --netG mamba_pfan \
    --netD basic \
    --direction AtoB \
    --dataset_mode aligned \
    --norm batch \
    --checkpoints_dir ./checkpoints  \
    --embed_dim 64 \
    --ndf 64 \
    --ngf 64   \
    --gpu_ids 0 \
    --batch_size 4 \
    --display_id -1 \
    --n_epochs 200 \
    --save_epoch_freq 5
