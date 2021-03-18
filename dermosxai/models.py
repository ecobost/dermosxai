""" Pytorch models. """
import torch
from torch import nn
#from torch.nn import functional as F

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


        

class VAE(nn.Module):
    def __init__(self, img_channels=3, encoder_kernels=(3, 3, 3), encoder_fmaps=(3, 32, 32, 32), 
                hidden_dims=16):
        super().__init__()
        
        # Define encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 16, kernel_size=7, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(), 
            nn.Linear(480, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2 * hidden_dims)
        )
        
        
        #TODO: Test broadcast decoder
        # Define decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 480, bias=False),         
            nn.BatchNorm1d(480),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (32, 3, 5)),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, bias=False, output_padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, bias=False, output_padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, bias=False, output_padding=(1, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, img_channels, kernel_size=7, stride=2, output_padding=1),
        )
        
        # Save params
        self.hidden_dims=hidden_dims
    
    def sample(self, z_params):
        """ Considers the first half od the dims as the mean ad the rest as the std."""
        return z_params[:, :self.hidden_dims]
    
    def forward(self, x):
        z_params = self.encoder(x)
        z = self.sample(z_params) 
        x_tilde = self.decoder(z)
        return x_tilde
    
    def init_parameters(self):
        layers = [*self.encoder, *self.decoder]
        init_conv(m for m in layers if isinstance(m, (nn.Conv2d, nn.Linear)))
        #init_conv([m for m in layers if isinstance(m, nn.ConvTranspose2d)], is_transposed=True)
        #init_bn(m for m in layers if isinstance(m, nn.BatchNorm2d))