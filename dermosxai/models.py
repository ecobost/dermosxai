""" Pytorch models. """
import torch
from torch import nn
from torch.nn import functional as F


def init_conv(modules, is_transposed=False):
    """ Initializes all module weights using He initialization and set biases to zero.
    
    Arguments:
        is_transposed (bool): Whether the modules are transposed convolutions. If so, 
            we use kaiming init with mode='fan_out'; this is correct for stride=dilation=1
            and approx. right for transposed convolutions with higher stride or dilation.
    """
    for module in modules:
        nn.init.kaiming_normal_(module.weight, mode='fan_out' if is_transposed else 'fan_in')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def init_bn(modules):
    """ Initializes all module weights to N(1, 0.1) and set biases to zero."""
    for module in modules:
        nn.init.normal_(module.weight, mean=1, std=0.1)
        nn.init.constant_(module.bias, 0)


class ConvEncoder(nn.Module):
    """ Encodes an 2d input into a single dimensional feature vector using convs.

    Downsampling is done by 4 x 4 convs with stride=2. It is recommended to use kernel
    sizes that are divisible by the stride to avoid checkerboard artifacts.

    Input: N x in_channels x 128 x 128
    Output: N x out_channels

    Arguments:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Size of the outputted feature vector.
    """
    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()

        # Create layers
        layers = [
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1,
                      padding_mode='reflect', bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=4),  # output is 1 x 1
            ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).squeeze(-1).squeeze(-1)

    def init_parameters(self):
        init_conv(m for m in self.layers if isinstance(m, nn.Conv2d))
        init_bn(m for m in self.layers if isinstance(m, nn.BatchNorm2d))

class UnPad(nn.Module):
    """ Inverse of padding. Drops some edge around an image. 
    
    Arguments:
        padding (int) Amount of padding to drop.
    """
    def __init__(self, padding=1):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return x[..., self.padding:-self.padding, self.padding:-self.padding]


class TransposeDecoder(nn.Module):
    """ Decodes a feture vector into an image using transposed convolutions.
    
    Upsampling is done by 4 x 4 convs with stride=2. It is recommended to use kernel sizes
    that are divisible by the stride to avoid checkerboard artifacts.
    
    Input: N x in_channels
    Output: N x out_channels x 128 x 128
    
    Arguments:
        in_channels (int): Size of the input feature vector.
        out_channels (int): Number of channels in the expected output image.
    """
    def __init__(self, in_channels=128, out_channels=3):
        super().__init__()

        # Create layers
        layers = [
            nn.ConvTranspose2d(in_channels, 128, kernel_size=4, bias=False), # 4 x 4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, bias=False),
            UnPad(1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, bias=False),
            UnPad(1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, bias=False),
            UnPad(1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2),
            UnPad(1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x.unsqueeze(-1).unsqueeze(-1))

    def init_parameters(self):
        init_bn(m for m in self.layers if isinstance(m, nn.BatchNorm2d))
        init_conv([m for m in self.layers if isinstance(m, nn.ConvTranspose2d)],
                  is_transposed=True)

class ResizeDecoder(nn.Module):
    """ Decodes a feture vector into an image using resized convolutions.

    Upsampling is done directly in each feature map using interpolation. Each upsampling
    is followed by a convolution (1x1 for the first two upsamplings, 3 x 3 afterwards).

    Initial upsampling from 1x1 to 4x4 is done with a transposed convolution. Otherwise,
    the same vector is repeated at all x, y positions and any kernel on this feature maps
    will produce the exact value for all x, y positions (as the inputs will be exactly the
    same everywhere) and all the way to the reconstruction all channels will hve the same
    value everywhere (i.e., it is just able t predict a single color). This symmetry could
    have also been broken byusing zero padding rather tha reflection padding but seems
    pretty ad hoc.

    Input: N x in_channels
    Output: N x out_channels x 128 x 128

    Arguments:
        in_channels (int): Size of the input feature vector.
        out_channels (int): Number of channels in the expected output image.
        mode (str): Usampling mode sent to nn.Upsample. Usually 'bilinear' or 'nearest'.
    """
    def __init__(self, in_channels=128, out_channels=3, mode='nearest'):
        super().__init__()

        # Create layers
        layers = [
            nn.ConvTranspose2d(in_channels, 128, kernel_size=4, bias=False), # 4 x 4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1, padding_mode='reflect')]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x.unsqueeze(-1).unsqueeze(-1))

    def init_parameters(self):
        init_conv(m for m in self.layers if isinstance(m, nn.Conv2d))
        init_bn(m for m in self.layers if isinstance(m, nn.BatchNorm2d))
        init_conv([m for m in self.layers if isinstance(m, nn.ConvTranspose2d)],
                  is_transposed=True)


class PixelShuffleDecoder(nn.Module):
    """ Decodes a feture vector into an image using pixel shuffling.
    
    Upsampling is done by creating upsampling_factor**2 * feature_maps feature maps and 
    then reordering them to have a num_features x upsampling_factor*h x 
    upsamplig_factor*w block
    
    Input: N x in_channels
    Output: N x out_channels x 128 x 128
    
    Arguments:
        in_channels (int): Size of the input feature vector.
        out_channels (int): Number of channels in the expected output image.
    """
    def __init__(self, in_channels=128, out_channels=3):
        super().__init__()

        # Create layers
        layers = [
            nn.Conv2d(128, 128 * 16, kernel_size=1),
            nn.PixelShuffle(4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64 * 4, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32 * 4, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32 * 4, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels * 4, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.PixelShuffle(2)
            ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x.unsqueeze(-1).unsqueeze(-1))

    def init_parameters(self):
        init_conv(m for m in self.layers if isinstance(m, nn.Conv2d))
        init_bn(m for m in self.layers if isinstance(m, nn.BatchNorm2d))


class VAE(nn.Module):
    """ VAE with a multivariate normal q(z|x) with diagonal covariance matrix.
    
    Arguments:
        img_channels (int): Number of channels in the input.
        hidden_dims (int): Number of variables in the intermediate representation. The 
            encoder predicts twice as many features, the second half are the log(std) 
            components.
        decoder (str): Decoder to use. One of "transposed", "resize" or "shuffle".
    """
    def __init__(self, img_channels=3, hidden_dims=128, decoder='transposed'):

        super().__init__()

        # Define encoder
        self.encoder = ConvEncoder(img_channels, hidden_dims * 2)

        # Define decoder
        if decoder == 'transposed':
            self.decoder = TransposeDecoder(hidden_dims, img_channels)
        elif decoder == 'resize':
            self.decoder = ResizeDecoder(hidden_dims, img_channels)
        elif decoder == 'shuffle':
            self.decoder = PixelShuffleDecoder(hidden_dims, img_channels)
        else:
            raise ValueError(f"Decoder '{decoder}' not recognized.")

        # Save params
        self.hidden_dims=hidden_dims
        self.sample_z = True # whether to sample z from q(z|x) or return the mean


    def encode(self, x):
        """Takes original image to predicted mu and sigma for the proposal distribution. 
        
        Arguments:
            x (torch.FloatTensor): Batch of images (N x C X H X W).
        
        Returns:
            mu (torch.FloatTensor): A (N x hidden_dims) vector with the predicted means.
            logsigma (torch.FloatTensor): A (N x hidden_dims) vector with the predicted 
                logstds. std = exp(logsigma)
        """
        # Resizing image to 128 x 128
        self.original_dims = x.shape[2:] # used for resizing
        resized_x = F.interpolate(x, (128, 128)) #, mode='bilinear', align_corners=False)

        # Encode image into u, sigma for gaussian q distribution
        q_params = self.encoder(resized_x)
        mu, logsigma = self._decode_gaussian_params(q_params)

        return mu, logsigma

    def _decode_gaussian_params(self, q_params):
        """ Takes the intermediate representation and transforms it into mean and logsigma 
        for the q(z|x) distribution.
        
        First half of the vector is the mean and the second part the logsigma.
        """
        mu = q_params[:, :self.hidden_dims]
        logsigma = q_params[:, self.hidden_dims:]
        return mu, logsigma

    def decode(self, z):
        """ Takes a hidden vector to image. """
        recons = self.decoder(z)
        resized_recons = F.interpolate(recons, self.original_dims)#, mode='bilinear', align_corners=False)

        return resized_recons

    def train(self, mode=True):
        super().train(mode)
        self.sample_z = mode  # disable sampling of z during evaluation (eval() calls train(False))

    def forward(self, x):
        """ Forward and return all intermediate values.
        
        Returns:
            q_params (tuple): Predicted params (mu, logsigma) for the gaussian q(z|x) 
                distribution.
            z (torch.Tensor): Sampled value from q(z|x) (with the predicted params).
            recons (torch.Tensor): Reconstructed image from the sample z.
        """
        mu, logsigma = self.encode(x)
        z = gaussian_sample(mu, torch.exp(logsigma)) if self.sample_z else mu
        recons = self.decode(z)
        return (mu, logsigma), z, recons

    def init_parameters(self):
        self.encoder.init_parameters()
        self.decoder.init_parameters()


# When doing the VAE where the intermediate features are shared, I can reuse the
# encoder/decoder and overall interface of this VAE (with a encode(), decode() and
# decode_q_params() function).



#TODO: Maybe move this functions to a diff dist_utils.py (where I can also put other sampling stuff)
import math


def gaussian_sample(mu, sigma):
    """ Sample a gaussian variable using the reparametrization trick.
    
    Arguments:
        mu (torch.Tensor): Mean.
        sigma (torch.Tensor): Standard deviation. Same size as mu.
    
    Returns:
        z (torch.Tensor): Samples. Same size as mean.
    """
    noise = torch.randn_like(mu)
    return mu + noise * sigma


def gaussian_kl(mu, logsigma):
    """ Analytically compute KL divergence between the multivariate gaussian defined by 
    the input params (assuming diagonal covariance matrix) and N(0, I). 
    
    Arguments:
        mu (torch.Tensor): Mean. Expected size (N x num_variables)
        logsigma (torch.Tensor): Natural logarithm of the standard deviation. Same size 
            as mean.
    
    Returns:
        kl (torch.Tensor): KL divergence for each example in the batch.
        
    Note:
        Same result as torch.distributions.kl_divergence(q, p)
    """
    #kl = ((sigma**2 + mu**2) / 2 - torch.log(sigma) - 0.5).sum(-1) # unstable
    kl = ((torch.exp(2 * logsigma) + mu**2)/ 2 - logsigma - 0.5).sum(-1)
    return kl


def gaussian_tc(mu, logsigma, dset_size=1, eps=1e-9):
    """Compute the TC term KL(q(z) || prod_i q(z_i)) using minibatch-weighted sampling.
    
    Reduces to 1/M sum_i^M log(sum_j^M q(z^(i) | x_j)) - log(NM) where z^(i) ~ q(z|x_i)
    
    Arguments:
        mu (torch.Tensor): Mean. Expected size (N x num_variables)
        logsigma (torch.Tensor): Natural logarithm of the standard deviation. Same size 
            as mean.
        dset_size (int): Number of examples in the dset. Use 1 to avoid the dset_size 
            having an effect in the tc estimation (batch size ofc still matters).
        eps (float): Small number to avoid division by zero.
        
    Returns:
        tc(float): Total correlation. Differentiable.
        
    Note: 
        Assumes q(z_i|x) is independent of q(z_j|x) for all i, j.
    """
    sigma = torch.exp(logsigma)

    # Sample z (one per example)
    z = gaussian_sample(mu, sigma) #if self.sample_z else mu

    # Compute log cross-probabilities logq(z^(i)|x_j) (i, j are rows and columns)
    lognorm_cons = logsigma + 0.5 * math.log(2 * math.pi)
    logxprob = -lognorm_cons - 0.5 * ((z[:, None] - mu) / (sigma + eps))**2

    # Estimate logq(z)
    logqz = torch.logsumexp(logxprob.sum(-1), dim=1).mean(0) - math.log(dset_size * len(z))
    logqz_i = torch.logsumexp(logxprob, dim=1).mean(0) - math.log(dset_size * len(z))

    # Calculate total correlation
    tc = logqz - logqz_i.sum()

    return tc



############################### CLASSIFICATION MODELS ###################################
from torchvision import models


class ResNetBase(nn.Module):
    """ A trainable ResNet50 module cropped after some number of layers (0-5).
        
    Arguments:
        num_blocks (int): Number of residual blocks the resulting ResNet is clipped at.
            Every resnet has 4 blocks each with diff number of convs (see Tab.1 in 
            He et al. 2015).
        pretrained (bool): Whether the Resnet is pretrained on ImageNet.
    
    Note:
        Blocks outputs are usually ReLUed. Outputs are also not avgpooled.
    """
    def __init__(self, num_blocks, pretrained=True):
        super().__init__()

        # Load resnet
        self.pretrained = pretrained
        self.resnet = models.resnet50(pretrained=pretrained)  # could be any other ResNet

        # Save num_blocks
        if num_blocks < 0 or num_blocks > 5:
            raise ValueError('num_blocks needs to be in [0, 5] range')
        self.num_blocks = num_blocks

        # Drop unnecesary modules (not necessary but makes network smaller when saving)
        if num_blocks < 5:
            del self.resnet.fc
            del self.resnet.avgpool
        if num_blocks < 4:
            del self.resnet.layer4
        if num_blocks < 3:
            del self.resnet.layer3
        if num_blocks < 2:
            del self.resnet.layer2
        if num_blocks < 1:
            del self.resnet.layer1

        # Find out number of channels in the final feature map (just for documentation)
        with torch.no_grad():
            self.eval()
            self.out_channels = self.forward(torch.randn(1, 3, 224, 224)).shape[1]
            self.train()

    def forward(self, x):
        """Copies the forward from resnet50 and crops it at desired depth"""
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        if self.num_blocks == 0:
            return x

        x = self.resnet.layer1(x)
        if self.num_blocks == 1:
            return x
        x = self.resnet.layer2(x)
        if self.num_blocks == 2:
            return x
        x = self.resnet.layer3(x)
        if self.num_blocks == 3:
            return x
        x = self.resnet.layer4(x)
        if self.num_blocks == 4:
            return x

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)

        return x

    def init_parameters(self):
        if self.pretrained:
            print('ResNetBase using pretrained weights for initialization')
        else:
            raise NotImplementedError


class MultiLinearHead(nn.Module):
    """Average pools input and applies different linear layers to the input.
    
    Arguments:
        in_channels (int): Number of feature maps in the input.
        out_channels (list of int): Number of output channels for each of the linear 
            layers.
    
    Note:
        The advantage of this module over a simple linear layer with sum(out_channels) 
        outputs is that each linear readout here has its own bias and it is easier to 
        later apply a softmax to each of this outputs.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        linear_layers = [nn.Linear(in_channels, oc) for oc in out_channels]
        self.linear_layers = nn.ModuleList(linear_layers)

        self.out_channels = out_channels

    def init_parameters(self):
        init_conv(self.linear_layers)

    def forward(self, x):
        x_flat = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1) if x.ndim == 4 else x
        return [ll(x_flat) for ll in self.linear_layers]


class ResNetPlusMultiLinear(nn.Module):
    """ ResNet with multiple linear readouts.
        
    Arguments:
        num_blocks (int): Number of resnet blocks for the base (0-5).
        pretrained_resnet (bool):  Wheher to load the pretrained version of the resnet.
        out_channels (list of int): Number of outputs for each linear readout.
    """
    def __init__(self, num_blocks, out_channels, pretrained_resnet=True):
        super().__init__()
        self.base = ResNetBase(num_blocks, pretrained=pretrained_resnet)
        self.head = MultiLinearHead(in_channels=self.base.out_channels,
                                    out_channels=out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        x_feats = self.base(x)
        out = self.head(x_feats)
        return out

    def init_parameters(self):
        self.base.init_parameters()
        self.head.init_parameters()


class SoftmaxPlusConcat(nn.Module):
    """ Helper module that takes the output of a multilinear head, applies a softmax to 
    each output and concatenates them to create a feature vector of probs.
    
    Input: List of N x d_i batches of logits.
    Output: N x sum_i(d_i) batch with concatenated probs.
    """
    def forward(self, x):
        probs = [F.softmax(x_i, -1) for x_i in x]
        return torch.cat(probs, -1)


class ConvNetBase(nn.Module):
    """ A small convnet architecture to work as a feature extractor.
    
    Arguments:
        in_channels (int): Expected input channels in the image.
        out_channels (int): Desired output channels
    """
    def __init__(self, in_channels=3, out_channels=128):
        super().__init__()

        # Create layers
        layers = [
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=2, padding=3,
                      padding_mode='reflect', bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1,
                      padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), ]
        self.layers = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x):
        return self.layers(x)

    def init_parameters(self):
        init_conv(m for m in self.layers if isinstance(m, nn.Conv2d))
        init_bn(m for m in self.layers if isinstance(m, nn.BatchNorm2d))


class JointWithLinearHead(nn.Module):
    """ Classifier with three components: a human attribute predictor, a feature extractor
    and a linear readout. 
    
    Attribute predictions are concatenated with the extracted image features and send 
    through a linear layer to obtain the predicted class.
    
    Arguments:
        abl (nn.Module): Attribute learner model: predicts a list of logits per attribute.
        extractor (nn.Module): Feature extractor: produces image features (N x C x H x W).
        out_channels (int): Number of output channels to produce.
    """
    def __init__(self, abl, extractor, out_channels):
        super().__init__()
        self.abl = abl
        self.extractor = extractor  # a convnet or resnet

        self.linear_human = nn.Linear(in_features=sum(abl.out_channels),
                                      out_features=out_channels)
        self.linear_convnet = nn.Linear(in_features=extractor.out_channels,
                                        out_features=out_channels, bias=False)
        self.out_channels = out_channels

    def forward(self, x):
        # Get human attributes
        logits_per_attr = self.abl(x)
        probs_per_attr = [F.softmax(lgt, -1) for lgt in logits_per_attr]
        self.human_features = torch.cat(probs_per_attr, dim=-1)

        # Get image features
        self.convnet_features = torch.mean(self.extractor(x), dim=(-1, -2))

        # Apply linear on top
        out = self.linear_human(self.human_features) + self.linear_convnet(
            self.convnet_features)

        return out

    def init_parameters(self, init_abl=False, init_extractor=True):
        if init_abl:
            self.abl.init_parameters()
        if init_extractor:
            self.extractor.init_parameters()
        init_conv([self.linear_human, self.linear_convnet])
