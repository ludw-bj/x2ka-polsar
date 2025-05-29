import torch
from .base_model import BaseModel
from . import networks
from torch.autograd import Variable


class PerceptualModel(BaseModel):
    """ This class implements the PerceptualLoss(2016) model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    PerceptualLoss paper: https://arxiv.org/abs/1603.08155
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # changing the default values
        parser.set_defaults(norm='batch', dataset_mode='aligned')
        if is_train:
            parser.add_argument('--lambda_style', type=float, default=1e5, help='weight for style loss')
            parser.add_argument('--lambda_content', type=float, default=1e0, help='weight for content loss')

        return parser

    def __init__(self, opt):
        """Initialize the PerceptualLoss class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['content', 'style', 'mse']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['Trans']
        else:  # during test time, only load TransNet
            self.model_names = ['Trans']

        # define networks
        self.netTrans = networks.define_Trans(opt.input_nc, opt.output_nc, opt.init_type, opt.init_gain, self.gpu_ids).type(opt.tensorType)
        if self.isTrain:
            self.netVgg = networks.Vgg16().type(opt.tensorType)
            # define loss functions
            self.criterionMSE = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(self.netTrans.parameters(), lr=opt.lr)
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device).type(self.opt.tensorType)
        self.real_B = input['B' if AtoB else 'A'].to(self.device).type(self.opt.tensorType)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netTrans(self.real_A)

    def backward(self):
        """Calculate loss"""
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


    def optimize_parameters(self):
        self.forward()                   # compute fake images

        self.set_requires_grad(self.netTrans, True)  # enable backprop for TransformNet
        self.set_requires_grad(self.netVgg, False)  # disable backprop for VGG16
        self.optimizer.zero_grad()     # set gradients to zero
        self.backward()                # calculate gradients
        self.optimizer.step()          # update weights
