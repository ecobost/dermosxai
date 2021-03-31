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


class AddCoords(nn.Module):
    """Add x, y channels to an image with [-1, 1] px coordinates.
    
    Arguments:
        height, width (int): Size of the mesh of coordinates to add.
    """
    def __init__(self, height=128, width=128):
        super().__init__()

        # Create mesh
        x = torch.linspace(-1, 1, width)
        y = torch.linspace(-1, 1, height)
        mesh = torch.meshgrid(y, x)
        mesh = torch.stack(mesh, 0)[None, ...]  # 1 x 2 x height x width
        self.register_buffer('mesh', mesh)

    def forward(self, x):
        tiled_mesh = torch.tile(self.mesh, (x.shape[0], 1, 1, 1))
        x_plus_coord = torch.cat([tiled_mesh, x], 1)
        return x_plus_coord


class BroadcastDecoder(nn.Module):
    """ Decodes a feture vector into an image using a spatial broadcast decoder
    (Watters et al., 2019).
    
    Tile the feature vector to the desired final size, add x-y coordinates as two extra 
    feature maps and then apply normal convolutions.
    
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
            nn.Upsample(128),
            AddCoords(128, 128),
            nn.Conv2d(in_channels + 2, 128, kernel_size=3, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1,
                      padding_mode='reflect'),
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
        decoder (str): Decoder to use. One of "transposed", "resize", "shuffle" or 
            "broadcast".
    """
    def __init__(self, img_channels=3, hidden_dims=128, decoder='resize'):

        super().__init__()

        # Define encoder
        self.encoder = ConvEncoder(img_channels, hidden_dims * 2)

        # Define decoder
        if decoder == 'transpose':
            self.decoder = TransposeDecoder(hidden_dims, img_channels)
        elif decoder == 'resize':
            self.decoder = ResizeDecoder(hidden_dims, img_channels)
        elif decoder == 'shuffle':
            self.decoder = PixelShuffleDecoder(hidden_dims, img_channels)
        elif decoder == 'broadcast':
            self.decoder = BroadcastDecoder(hidden_dims, img_channels)
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
            sigma (torch.FloatTensor): A (N x hidden_dims) vector with the predicted stds.
        """
        # Resizing image to 128 x 128
        self.original_dims = x.shape[2:] # used for resizing
        resized_x = F.interpolate(x, (128, 128), mode='bilinear', align_corners=False)

        # Encode image into u, sigma for gaussian q distribution
        q_params = self.encoder(resized_x)
        mu, sigma = self._decode_gaussian_params(q_params)

        return mu, sigma

    def _decode_gaussian_params(self, q_params):
        """ Takes the intermediate representation and transforms it into mean and sigma 
        for the q(z|x) distribution.
        
        First half of the vector is the mean and the second part the logsigma.
        """
        mu = q_params[:, :self.hidden_dims]
        sigma = torch.exp(q_params[:, self.hidden_dims:])
        return mu, sigma

    def decode(self, z):
        """ Takes a hidden vector to image. """
        recons = self.decoder(z)
        resized_recons = F.interpolate(recons, self.original_dims, mode='bilinear',
                                       align_corners=False)
        return resized_recons

    def train(self, mode=True):
        super().train(mode)
        self.sample_z = mode  # disable sampling of z during evaluation (eval() calls train(False))

    def forward(self, x):
        """ Forward and return all intermediate values.
        
        Returns:
            q_params (tuple): Predicted params (mu, sigma) for the gaussian q(z|x) 
                distribution.
            z (torch.Tensor): Sampled value from q(z|x) (with the predicted params).
            recons (torch.Tensor): Reconstructed image from the sample z.
        """
        mu, sigma = self.encode(x)
        z = gaussian_sample(mu, sigma) if self.sample_z else mu
        recons = self.decode(z)
        return (mu, sigma), z, recons

    def init_parameters(self):
        self.encoder.init_parameters()
        self.decoder.init_parameters()


# When doing the VAE where the intermediate features are shared, I can reuse the
# encoder/decoder and overall interface of this VAE (with a encode(), decode() and
# decode_q_params() function).





#TODO: Maybe move this functions to a diff dist_utils.py (where I can also put other sampling stuff)
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


def gaussian_KL(mu, sigma):
    """ Analytically compute KL divergence between the multivariate gaussian defined by 
    the input params (assuming diagonal covariance matrix) and N(0, I). 
    
    Arguments:
        mu (torch.Tensor): Mean. Expected size (N x num_variables)
        sigma (torch.Tensor): Standard deviation. Same size as mean.
    
    Returns:
        kl (torch.Tensor): KL divergence for each example in the batch.
    """
    kl = ((sigma**2 - 1).sum(-1) + (mu**2).sum(-1)) / 2 - torch.log(sigma).sum(-1)
    return kl


#TODO: Actually implement this
def approx_kl(p, q, z):
    """ Compute a MonteCarlo approximation of the KL(q|p): 1/n * (log(q(z)) - log(p(z))) 
    
    Arguments:
        p (torch.distributions): The distribution of the prior p(z).
        q (torch.distributions): The distribution of q(z|x) distribution.
        z (torch.Tensor): Samples from q (N x num_variables). 
    
    Returns:
        kl (torch.Tensor): KL divergence estimate.
    """
    raise NotImplementedError('Test and make sure this is alright!')
    kl = (p.log_prob(z) - q.log_prob(z)).sum(axis=-1).mean(0)
    return kl
