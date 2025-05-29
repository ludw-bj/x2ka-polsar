set -ex
python train.py --dataroot ./datasets/x2ka --name x2ka_pix2pix --model pix2pix --netG unet_256 --pretrain_G
python train.py --dataroot ./datasets/x2ka --name x2ka_pix2pix --model pix2pix --netG unet_256 --pretrain_D --save_epoch_freq 1
python train.py --dataroot ./datasets/x2ka --name x2ka_pix2pix --model pix2pix --netG unet_256