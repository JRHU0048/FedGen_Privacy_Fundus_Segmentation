import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import math
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.models as models 
import cv2
import torch.nn as nn
from scipy.optimize import minimize
from torch.cuda.amp import autocast, GradScaler

# ==============================================
# Define the loss class of Fourier transform in high-frequency space
# ==============================================

class HighFrequencyFFTLoss(torch.nn.Module):
    def __init__(self):
        super(HighFrequencyFFTLoss, self).__init__()

    def forward(self, real_image, generated_image):
        """
        Calculate the fast Fourier transform loss of real images and generated images in the high-frequency space.

        :param real_image: Real image tensor, with shapes of (batch size, channels, height, width)
        :param generated_image: Generate image tensors with shapes of (batch size, channels, height, width)
        :return: Fourier transform loss value in high-frequency space
        """
        batch_size, channels, height, width = real_image.size()

        real_image_reshaped = real_image.view(batch_size * channels, height, width)
        generated_image_reshaped = generated_image.view(batch_size * channels, height, width)

        # Perform the fast Fourier transform on the real image and the generated image
        real_image_fft = torch.fft.fft2(real_image_reshaped)
        generated_image_fft = torch.fft.fft2(generated_image_reshaped)

        # Move the zero-frequency component to the center of the spectrum to facilitate the subsequent separation of high-frequency and low-frequency components
        real_image_fft_shifted = torch.fft.fftshift(real_image_fft)
        generated_image_fft_shifted = torch.fft.fftshift(generated_image_fft)

        real_image_fft_shifted = real_image_fft_shifted.view(batch_size, channels, height, width)
        generated_image_fft_shifted = generated_image_fft_shifted.view(batch_size, channels, height, width)
        
        center_height, center_width = height // 2, width // 2
        u_coord = torch.arange(height).unsqueeze(1).repeat(1, width).unsqueeze(0).unsqueeze(0).to(real_image_fft_shifted.device)
        v_coord = torch.arange(width).unsqueeze(0).repeat(height, 1).unsqueeze(0).unsqueeze(0).to(real_image_fft_shifted.device)
        D_uv = torch.sqrt((u_coord - center_height)**2 + (v_coord - center_width)**2)

        D_0 = min(height, width) / 4
        n = 2
        butterworth_filter = 1 / (1+(D_0 / D_uv)**(2*n))

        # Apply the filter to the spectrum to obtain the high-frequency part
        real_image_fft_high_freq = real_image_fft_shifted * butterworth_filter
        generated_image_fft_high_freq = generated_image_fft_shifted * butterworth_filter

        # Calculate the sum of the squares of the differences of the high-frequency components (first take the modulus of the complex number)
        diff = real_image_fft_high_freq - generated_image_fft_high_freq
        diff_magnitude = torch.abs(diff)
        loss = torch.mean(torch.sum(diff_magnitude ** 2))

        return loss

# ======================================
class MobileNetFeatureExtractor_1(nn.Module):
    def __init__(self, selected_layers):
        super(MobileNetFeatureExtractor_1, self).__init__()
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
        self.selected_layers = selected_layers
        self.layer_names = {
            1: 'layer1', 3: 'layer2', 6: 'layer3', 13:'layer4', 15: 'layer5'
        }

    def forward(self, x):
        features = []
        for idx, layer in enumerate(self.mobilenet):
            x = layer(x)
            if idx in self.selected_layers:
                features.append(x)
        return features

class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_model', type=str, default="ppfg", choices='(ppfg, CUT, cut, FastCUT, fastcut)')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=0.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--lambda_fft', type=float, default=  1e-7, help='weight for high frequency FFT loss')
        parser.add_argument('--lambda_feature_matching', type=float,default= 2  ,help='weight for MobileNet loss')
        parser.add_argument('--lambda_shallow_feature_matching', type=float,default= 1 ,help='weight for MobileNet loss')
        parser.add_argument('--lambda_deep_feature_matching', type=float,default= 5 ,help='weight for MobileNet loss')
        parser.set_defaults(pool_size=0)  # no image pooling
        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        self.scaler = GradScaler() 

        #================================================================
        self.mobilenet_feature_extractor_1 = MobileNetFeatureExtractor_1(selected_layers=[1, 3, 6, 13, 15]).to(self.device)
        for param in self.mobilenet_feature_extractor_1.parameters():
            param.requires_grad = False 

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            self.criterionHighFreqFFT = HighFrequencyFFTLoss()
            self.criterionFeatureMatching = torch.nn.L1Loss()

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        
        with torch.amp.autocast(device_type='cuda'):
            self.loss_D = self.compute_D_loss()
        
        self.scaler.scale(self.loss_D).backward()
        self.scaler.step(self.optimizer_D)

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        
        with torch.amp.autocast(device_type='cuda'):
            self.loss_G = self.compute_G_loss()
        
        self.scaler.scale(self.loss_G).backward()
        self.scaler.step(self.optimizer_G)
        if self.opt.netF == 'mlp_sample':
            self.scaler.step(self.optimizer_F)
        
        self.scaler.update()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def forward(self):
        # """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        with torch.amp.autocast(device_type='cuda'):
            fake = self.fake_B.detach()
            pred_fake = self.netD(fake)
            self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
            self.pred_real = self.netD(self.real_B)
            loss_D_real = self.criterionGAN(self.pred_real, True)
            self.loss_D_real = loss_D_real.mean()
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        with torch.amp.autocast(device_type='cuda'):
            # GAN loss
            if self.opt.lambda_GAN > 0.0:
                pred_fake = self.netD(self.fake_B)
                self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
            else:
                self.loss_G_GAN = 0.0

            # NCE loss
            if self.opt.lambda_NCE > 0.0:
                self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
            else:
                self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

            if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
                self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
                loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
            else:
                loss_NCE_both = self.loss_NCE

            # Calculate the loss of high-frequency Fourier transform
            self.loss_G_HighFreqFFT = self.criterionHighFreqFFT(self.real_A, self.fake_B) * self.opt.lambda_fft

            # Calculate the feature matching loss of MobileNet
            real_features = self.mobilenet_feature_extractor_1(self.real_A)
            fake_features = self.mobilenet_feature_extractor_1(self.fake_B)
            self.loss_G_FeatureMatching = 0.0
            for real_feat, fake_feat in zip(real_features, fake_features):
                self.loss_G_FeatureMatching += self.criterionFeatureMatching(real_feat, fake_feat)

            self.loss_G_FeatureMatching *= self.opt.lambda_feature_matching

            #=================================== MGDA =======================================
            self.loss_G = self.compute_mgda_loss()

        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def compute_mgda_loss(self):
    
        losses = [self.loss_G_GAN, self.loss_NCE, self.loss_G_FeatureMatching, self.loss_G_HighFreqFFT] 
        grads = []
        param_shapes = []  
        param_sizes = []   

        for param in self.netG.parameters():
            param_shapes.append(param.shape)
            param_sizes.append(param.numel())

        # Calculate the gradient of each loss
        for loss in losses:
            with torch.amp.autocast(device_type='cuda'):
                loss.backward(retain_graph=True)
            grad = torch.cat([p.grad.view(-1) for p in self.netG.parameters()])  # Obtain the gradient
            grads.append(grad)
            self.netG.zero_grad()  # Clear the gradient

        # Define the objective function (minimize the weighted sum of gradients)
        def objective(weights):
            total_grad = torch.zeros_like(grads[0])
            for w, grad in zip(weights, grads):
                total_grad = total_grad + w * grad  
            return torch.norm(total_grad).item()

        # Define the constraint condition (the sum of the weights is 1)
        constraints = ({'type': 'eq', 'fun': lambda w: sum(w) - 1})

        # Initial weight
        w0 = [0.25, 0.25, 0.25, 0.25]

        # Solve convex optimization problems
        res = minimize(objective, w0, method='SLSQP', constraints=constraints, options={'maxiter': 10})
        weights = res.x

        # Update the model parameters using the weighted gradient
        total_grad = torch.zeros_like(grads[0])
        for w, grad in zip(weights, grads):
            total_grad = total_grad + w * grad 

        # Split total grad into the gradients of each parameter
        start = 0
        for param, shape, size in zip(self.netG.parameters(), param_shapes, param_sizes):
            end = start + size
            param_grad = total_grad[start:end].reshape(shape) 
            if param.grad is None:
                param.grad = param_grad.clone() 
            else:
                param.grad = param.grad + param_grad 
            start = end

        # Calculate the optimized loss
        optimized_loss = sum(w * loss for w, loss in zip(weights, losses))
        return optimized_loss