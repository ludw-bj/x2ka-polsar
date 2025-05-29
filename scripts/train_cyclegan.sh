set -ex
python train.py --dataroot ./datasets/x2ka --name x2ka_cyclegan --model cycle_gan --no_dropout
