"""training script:
- train options: options/base_options.py and options/train_options.py
- initiate or resume training from '--continue_train'
- during training: visualize/save the images, print/save the loss plot, and save models to '--checkpoints_dir'.

Example:
    Train a Perceptual model:
        python train.py --dataroot ./datasets/x2ka --name x2ka_perceptual --model perceptual
    Train a Pix2Pix model:
        python train.py --dataroot ./datasets/x2ka --name x2ka_pix2pix --model pix2pix --netG unet_256 --pretrain_G
        python train.py --dataroot ./datasets/x2ka --name x2ka_pix2pix --model pix2pix --netG unet_256 --pretrain_D --save_epoch_freq 1
        python train.py --dataroot ./datasets/x2ka --name x2ka_pix2pix --model pix2pix --netG unet_256
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/x2ka --name x2ka_cyclegan --model cycle_gan --no_dropout

"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.display_id = 1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        
        for i, data in enumerate(dataset):  # inner loop within one epoch          
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            if opt.save_grad2log or opt.display_grad:
                model.optimize_parameters_debug(idx = int(total_iters/opt.batch_size), epoch = epoch, idx_epoch = epoch_iter)
            else:
                model.optimize_parameters(idx = total_iters)

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                        
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        model.update_learning_rate()    # update learning rates at the end of every epoch.

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
