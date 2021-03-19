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


class UnPad(nn.Module):
    """Inverse of padding."""
    def __init__(self, padding=1):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return x[..., self.padding:-self.padding, self.padding:-self.padding]

class TileAndAddCoords(nn.Module):
    def __init__(self, size=128):
        super().__init__()
        # Create mesh
        x = torch.linspace(-1, 1, size)
        y = torch.linspace(-1, 1, size)
        mesh = torch.meshgrid(y, x)
        self.mesh = torch.stack(mesh, 0)[None, ...]  # 1 x 2 x size x size
        self.mesh = self.mesh.cuda()

        self.size = size

    def forward(self, x):
        tiled = torch.tile(x[..., None, None], (1, 1, self.size, self.size))
        tiled_mesh = torch.tile(self.mesh, (x.shape[0], 1, 1, 1))
        tiled_plus_coord = torch.cat([tiled_mesh, tiled], 1)
        return tiled_plus_coord



class VAE(nn.Module):
    def __init__(self, img_channels=3, encoder_kernels=(3, 3, 3), encoder_fmaps=(3, 32, 32, 32),
                hidden_dims=16):
        super().__init__()

        # Define encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 16, kernel_size=7, stride=2, padding=3, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, padding_mode='reflect', stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(512, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2 * hidden_dims)
        )


        #TODO: Test broadcast decoder
        # Define decoder


        # # Trasnposed convs
        # self.decoder = nn.Sequential(
        #     nn.Linear(hidden_dims, 128, bias=False),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 512, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Unflatten(1, (32, 4, 4)),
        #     nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, bias=False),
        #     UnPad(1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, bias=False),
        #     UnPad(1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, bias=False),
        #     UnPad(1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, bias=False),
        #     UnPad(1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(16, img_channels, kernel_size=8, stride=2),
        #     UnPad(3),
        # )
        # # kernel size needs to be divisible by stride to avoid artifacts
        # # and we need to drop the padding afterwards



        # # Resize convolution (i.e., upsample and then normal convs)
        # self.decoder = nn.Sequential(
        #     nn.Linear(hidden_dims, 128, bias=False),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 512, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Unflatten(1, (32, 4, 4)),
        #     nn.Upsample(scale_factor=2),#, mode='bilinear', align_corners=False),
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1, padding_mode='reflect',
        #               bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2),#, mode='bilinear', align_corners=False),
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1, padding_mode='reflect',
        #                        bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2),#, mode='bilinear', align_corners=False),
        #     nn.Conv2d(32, 16, kernel_size=3, padding=1, padding_mode='reflect',
        #                        bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2),#, mode='bilinear', align_corners=False),
        #     nn.Conv2d(16, 16, kernel_size=3, padding=1, padding_mode='reflect',
        #                        bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2),#, mode='bilinear', align_corners=False),
        #     nn.Conv2d(16, img_channels, kernel_size=7, padding=3, padding_mode='reflect'),
        # )

        # # Pixel shuffle upsampling
        # self.decoder = nn.Sequential(
        #     nn.Linear(hidden_dims, 128, bias=False),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 512, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Unflatten(1, (32, 4, 4)),
        #     nn.Conv2d(32, 32 * 4, kernel_size=3, padding=1,
        #                        padding_mode='reflect', bias=False),
        #     nn.PixelShuffle(2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32 * 4, kernel_size=3, padding=1,
        #                        padding_mode='reflect', bias=False),
        #     nn.PixelShuffle(2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 16*4, kernel_size=3, padding=1, padding_mode='reflect',
        #                        bias=False),
        #     nn.PixelShuffle(2),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 16*4, kernel_size=3, padding=1, padding_mode='reflect',
        #               bias=False),
        #     nn.PixelShuffle(2),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, img_channels*4, kernel_size=7, padding=3, padding_mode='reflect'),
        #     nn.PixelShuffle(2),
        # )

        # Broadcast decoder (i.e., upsample and then normal convs)
        self.decoder = nn.Sequential(
            TileAndAddCoords(128),
            nn.Conv2d(hidden_dims + 2, 32, kernel_size=3, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, padding_mode='reflect',
                               bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, padding_mode='reflect',
                               bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, padding_mode='reflect',
                               bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, img_channels, kernel_size=7, padding=3, padding_mode='reflect'),
        )



        # Save params
        self.hidden_dims=hidden_dims
        self.resize_images = (128, 128) # TODO: receive this as input?

    def sample(self, z_params):
        """ Considers the first half od the dims as the mean ad the rest as the std."""
        return z_params[:, :self.hidden_dims]

        #TODO: Add so that when the model is put in eval the sampling only returns the mean, or maybe add a diff variable for that
        # if self.eval:
        #     pass

    def forward(self, x):
        original_dims = x.shape[2:]
        resized_x = F.interpolate(x, (128, 128), mode='bilinear', align_corners=False)


        z_params = self.encoder(resized_x)
        z = self.sample(z_params)
        x_tilde = self.decoder(z)

        final_recon = F.interpolate(x_tilde, original_dims, mode='bilinear',
                                    align_corners=False)


        return final_recon

    def init_parameters(self):
        pass
        #layers = [*self.encoder, *self.decoder]
        #init_conv(m for m in layers if isinstance(m, (nn.Conv2d, nn.Linear)))
        #init_conv([m for m in layers if isinstance(m, nn.ConvTranspose2d)], is_transposed=True)
        #init_bn(m for m in layers if isinstance(m, nn.BatchNorm2d))