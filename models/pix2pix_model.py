import torch
import os
from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.add_argument('--pretrain_G', action='store_true', help='Pretraining G?')
        parser.add_argument('--prepath_G', type=str, default='', help='path to pretrained G-net; default no pretrain')
        # parser.add_argument('--prepath_G', type=str, default='x2ka_pix2pix_pretrain_G', help='path to pretrained G-net')
        parser.add_argument('--lambda_style', type=float, default=1e5, help='weight for style loss')
        parser.add_argument('--lambda_content', type=float, default=1e0, help='weight for content loss')

        parser.add_argument('--pretrain_D', action='store_true', help='Pretraining D?')
        parser.add_argument('--prepath_D', type=str, default='', help='path to pretrained D-net; default no pretrain')
        # parser.add_argument('--prepath_D', type=str, default='x2ka_pix2pix_pretrain_D', help='path to pretrained D-net')

        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        parser.add_argument('--use_statLoss', action='store_true', help='whether or not to use statistical loss')
        parser.add_argument('--use_L1Loss', action='store_true', help='whether or not to use L1 loss')

        # parameters for calculating statistical loss
        parser.add_argument('--patch_size', type=int, default=7, help='size of the sliding window (patch) used to compute local statistical properties over the image.')
        parser.add_argument('--stride', type=int, default=2, help='stride of the sliding window used during patch-based statistical computation.')
        parser.add_argument('--statistical_mode', type=str, default='mean,LP2', help='Comma-separated list of statistical metrics to compute within each patch.')
        parser.add_argument('--loss_mode', type=str, default='L1', help='Distance metric used to compare statistical properties between patches. [L1 | mse]')
        
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss')
            parser.add_argument('--lambda_L1', type=float, default=1e2, help='weight for L1 loss')
            parser.add_argument('--lambda_stat', type=float, default=1e2, help='weight for statistical loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # Get Training Phase from opt
        self.pretrain_G = opt.pretrain_G
        self.pretrain_D = opt.pretrain_D

        # Training Phase: pretrain G-Net
        if self.pretrain_G:
            # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
            self.loss_names = ['content', 'style', 'mse']
            # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
            self.visual_names = ['real_A', 'fake_B', 'real_B']
            # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
            self.model_names = ['G']
            # define networks (initiate generator)
            self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            # define loss functions
            self.netVgg = networks.Vgg16().type(opt.tensorType)
            self.criterionMSE = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
        # Training Phase: pretrain D-Net
        elif self.pretrain_D:
            # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
            self.loss_names = ['D_real', 'D_fake']
            # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
            self.model_names = ['D']
            # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
            self.visual_names = []
            # define networks (load generator, initiate discriminator)
            self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            load_suffix = '%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            load_filename = '%s_net_%s.pth' % (load_suffix, 'G')
            load_path = os.path.join(opt.checkpoints_dir, opt.prepath_G, load_filename)
            net = getattr(self, 'netG')
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print('loading the model from %s' % load_path)
            # if you are using PyTorch newer than 0.4 (e.g., built from GitHub source), you can remove str() on self.device
            state_dict = torch.load(load_path, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)
            # define networks (initiate discriminator)
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # define loss functions
            self.criterionGAN_D = networks.GANLoss(gan_mode = opt.gan_mode, target_real_label=1.0, target_fake_label=0.0).to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr * 4, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)
        # Training Phase: GAN (G & D)
        else:
            # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
            self.use_statLoss = opt.use_statLoss
            self.use_L1Loss = opt.use_L1Loss
            self.loss_names = ['D_real', 'D_fake', 'G_GAN']
            if self.use_statLoss:
                self.loss_names.append('G_stat')
            if self.use_L1Loss:
                self.loss_names.append('G_L1')
            # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
            self.visual_names = ['real_A', 'fake_B', 'real_B']
            # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
            if self.isTrain:
                self.model_names = ['G', 'D']
            else:  # during test time, only load G
                self.model_names = ['G']
            # define networks (both generator and discriminator)
            self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

            if self.isTrain:
                # define networks
                self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
                if not opt.prepath_G == '': # use pretrained_G
                    # load G
                    load_suffix = '%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
                    load_filename = '%s_net_%s.pth' % (load_suffix, 'G')
                    load_path = os.path.join(opt.checkpoints_dir, opt.prepath_G, load_filename)
                    net = getattr(self, 'netG')
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    print('loading the model from %s' % load_path)
                    # if you are using PyTorch newer than 0.4 (e.g., built from GitHub source), you can remove str() on self.device
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata
                        # patch InstanceNorm checkpoints prior to 0.4
                        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                            self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                        net.load_state_dict(state_dict)
                if not opt.prepath_D == '': # use pretrained_D
                    # load D
                    load_suffix = '%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
                    load_filename = '%s_net_%s.pth' % (load_suffix, 'D')
                    load_path = os.path.join(opt.checkpoints_dir, opt.prepath_D, load_filename)
                    net = getattr(self, 'netD')
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    print('loading the model from %s' % load_path)
                    # if you are using PyTorch newer than 0.4 (e.g., built from GitHub source), you can remove str() on self.device
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata
                        # patch InstanceNorm checkpoints prior to 0.4
                        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                            self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                        net.load_state_dict(state_dict)

                # define loss functions
                self.criterionGAN_G = networks.GANLoss(gan_mode = opt.gan_mode, target_real_label=1.0, target_fake_label=0.0).to(self.device)
                self.criterionGAN_D = networks.GANLoss(gan_mode = opt.gan_mode, target_real_label=1.0, target_fake_label=0.0).to(self.device)
                # self.criterionGAN_G = networks.GANLoss(gan_mode = opt.gan_mode, target_real_label=0.9, target_fake_label=0.1).to(self.device)
                # self.criterionGAN_D = networks.GANLoss(gan_mode = opt.gan_mode, target_real_label=0.9, target_fake_label=0.1).to(self.device)
                if self.use_statLoss:
                    self.criterionStat = networks.StatisticalLoss(opt.statistical_mode, opt.loss_mode, opt.patch_size, opt.stride).to(self.device)
                self.criterionL1 = torch.nn.L1Loss()
                # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                # self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr * 4, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_G)
                self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
    
    def backward_pretrainG(self):
        """Calculate loss for G-net pretraining"""
        # get vgg features
        A_features = self.netVgg(self.real_A)
        B_features = self.netVgg(self.real_B)
        fake_features = self.netVgg(self.fake_B)

        # calculate style loss
        fake_gram = [networks.gram(fmap) for fmap in fake_features]
        B_gram = [networks.gram(fmap) for fmap in B_features]
        style_loss = 0.0
        for j in range(4):
            style_loss += self.criterionMSE(fake_gram[j], B_gram[j])
        self.loss_style = self.opt.lambda_style * style_loss

        # calculate content loss (h_relu_2_2)
        self.loss_content = self.opt.lambda_content * self.criterionMSE(A_features[1], fake_features[1])

        # calculate mse loss
        self.loss_mse = self.criterionMSE(self.real_B, self.fake_B)

        # total loss
        self.total_loss = self.loss_style + self.loss_content

        # combine loss and calculate gradients
        self.total_loss.backward()

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN_D(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN_D(pred_real, True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * self.opt.lambda_GAN
        self.loss_D.backward() ##

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN_G(pred_fake, True)
        self.loss_G = self.loss_G_GAN * self.opt.lambda_GAN

        # Second, G(A) = B
        if self.use_L1Loss:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
            self.loss_G += self.loss_G_L1     

        # Third, share same statistical characteristics
        if self.use_statLoss:
            self.loss_G_stat = self.criterionStat(self.fake_B, self.real_B) * self.opt.lambda_stat
            self.loss_G += self.loss_G_stat

        # calculate gradients
        self.loss_G.backward()

    def optimize_parameters(self, idx):
        if self.pretrain_G:
            self.forward()                   # compute fake images
            # update G
            self.set_requires_grad(self.netG, True)  # enable backprop for TransformNet
            self.set_requires_grad(self.netVgg, False)  # disable backprop for VGG16
            self.optimizer_G.zero_grad()     # set gradients to zero
            self.backward_pretrainG()                # calculate gradients
            self.optimizer_G.step()          # update weights
        elif self.pretrain_D:
            self.forward()                   # compute fake images: G(A)
            # update D
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.set_requires_grad(self.netG, False)  # disable backprop for G
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
        else:           
            self.forward()                   # compute fake images: G(A)
            # update D
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
            # update G
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G()                   # calculate graidents for G
            self.optimizer_G.step()             # udpate G's weights
    
    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
