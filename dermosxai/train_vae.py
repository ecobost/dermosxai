""" Code to train VAE models. """
import copy
import time
from os import path

import torch
import wandb
from torch import optim
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils import data
from torchvision import transforms

from dermosxai import datasets, models, utils


def get_hyperparams():
    """ Returns a list of hyperparams to run."""
    # Test lr and alpha
    return [  #{'batch_size': 64, 'beta_weight': 0}, 
        {'batch_size': 128, 'beta_weight': 0},
        {'augmentation': 'full', 'beta_weight': 0},
        {'learning_rate': 0.001, 'beta_weight': 0},
        {'weight_decay': 0.001, 'beta_weight': 0}, ]


    # Test bigger aarchitectures
    # #Test:
    #     hyperparams (lr and alpha)
    #     transform (full)
    #     batch_size (128, 64, 32)
    #     decoder
    #     beta_weight
    #     loss function

def train_all():
    for params in get_hyperparams():
        train(**params)

def train_one(param_id):
    """ Utility function to run one hyperparam. Used to coordinate diff runs in the 
    cluster.
    
    Arguments:
        param_id: Which hyperparam set to train
    """
    all_params = get_hyperparams()
    selected_params = all_params[param_id]
    return train(**selected_params)


def train(decoder='resize', seed=19, batch_size=64, learning_rate=0.001, weight_decay=0.01,
          num_epochs=200, decay_epochs=5, lr_decay=0.1, stopping_epochs=25,
          loss_function='beta-vae', beta_weight=1, augmentation='flips'):
    """ Trains a VAE with ADAM and early stopping.
    
    Arguments:
        decoder(string): Type of decoder architecture for the VAE. One of "resize", 
            "transposed", "shuffled" and "broadcast". See models.py for details.
        seed(int): Random seed for torch and numpy
        batch_size (int): Batch size.
        learning_rate (float): Initial learning rate for the optimizer.
        weight_decay (float): Weight for the l2 regularization.
        num_epochs (int): Maximum number of epochs to run.
        decay_epochs (int): Number of epochs to wait before decreasing learning rate if 
            validation mse has not improved.
        lr_decay (float): Factor multiplying learning rate when decaying.
        stopping_epochs (int): Early stop training after this number of epochs without an 
            improvement in validation mse.
        loss_function (str): Loss function to optimize. One of "beta-vae" or "tcvae".
        beta_weight (float): Weight for the disentanglement term of the loss (KL or TC 
            depending on the type).
        augmentation (str): How to augment the training images. One of 'flips' or 'full'.
            'flips': Horizontal and vertical flips.
            'full': Add an affine transform (rotations, translations, shear and scale) and 
                color jittering (contrast, brightness, saturations and hue).
    
    Returns: 
        model (nn.Module): The trained pytorch module.
    """
    # Log wandb hyperparams
    hyperparams = {
        'decoder': decoder, 'seed': seed, 'batch_size': batch_size,
        'learning_rate': learning_rate, 'weight_decay': weight_decay,
        'num_epochs': num_epochs, 'decay_epochs': decay_epochs, 'lr_decay': lr_decay,
        'stopping_epochs': stopping_epochs, 'loss_function': loss_function,
        'beta_weight': beta_weight, 'augmentation': augmentation}
    wandb.init(project='dermosxai_vae', group='ham10K-only', config=hyperparams,
               tags=['no-zsampling'])  #mode='disabled')

    # Set random seed
    torch.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True

    # Get datasets
    utils.tprint('Loading dsets')
    train_dset = datasets.HAM10000('train')
    val_dset = datasets.HAM10000('val')

    # Add transforms
    img_mean, img_std = train_dset.img_mean, train_dset.img_std
    if augmentation == 'flips':
        extra_transforms = []
    elif augmentation == 'full':
        extra_transforms = [
            transforms.RandomAffine(degrees=90, shear=20, scale=(1, 1.4),
                                    translate=(0.03, 0.03), fill=list(img_mean / 255)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        ]
    else:
        raise ValueError(f'Augmentation {augmentation} is not a valid choice')
    train_transform = transforms.Compose([
        transforms.ToTensor(), # changes range to [0, 1]
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        *extra_transforms,
        transforms.Normalize(img_mean/255, img_std/255),
    ])
    train_dset.transform = train_transform

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(img_mean/255, img_std/255),
    ])
    val_dset.transform = val_transform

    # Create dloaders
    train_dloader = data.DataLoader(train_dset, batch_size=batch_size,
                                    shuffle=True, num_workers=4)
    val_dloader = data.DataLoader(val_dset, batch_size=128, num_workers=4)


    # Get model
    utils.tprint('Loading model')
    model = models.VAE(decoder=decoder)
    model.init_parameters()
    model.train()
    model.cuda()

    # Declare optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_decay,
                                               patience=decay_epochs, verbose=True)

    # Intialize variables for early stopping
    best_model = copy.deepcopy(model).cpu()
    best_epoch = 0
    best_nll = float('inf')

    # Train
    start_time = time.time()  # in seconds
    for epoch in range(1, num_epochs + 1):
        utils.tprint(f'Epoch {epoch}:')

        # Record learning rate
        wandb.log({'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']})

        # Loop over training set
        for batch_i, (images, _) in enumerate(train_dloader):
            # Zero the gradients
            model.zero_grad()

            # Move variables to GPU
            images = images.cuda()




            #TODO: DELETE
            model.sample_z = False



            # Forward
            q_params, _, recons = model(images)

            # Compute loss
            nll = ((recons - images)**2).sum(dim=(1, 2, 3)).mean()
            if loss_function == 'beta-vae':
                kl = models.gaussian_KL(*q_params).mean()
                loss = nll + beta_weight * kl
                wandb.log({'epoch': epoch, 'batch': batch_i, 'nll': nll.item(),
                          'kl': kl.item(), 'loss': loss.item()})

                if batch_i % 10 == 0:
                    utils.tprint(f'Training loss {loss.item():.5f}',
                                 f'(MSE: {nll.item():.5f}, KL: {kl.item():.5f})')
            elif loss_function == 'tcvae':
                #TODO: Maybe refactor to a _compute_loss() function
                raise NotImplementedError("this ain't it chief")
            else:
                raise ValueError(f'Loss function {loss_function} is not a valid option.')

            # Check for divergence
            if torch.isnan(loss) or torch.isinf(loss):
                wandb.run.summary['diverged'] =  True
                wandb.run.summary['best_epoch'] = best_epoch
                wandb.run.summary['best_val_nll'] = best_nll
                wandb.run.finish()
                raise ValueError('Training loss diverged')

            # Backprop
            loss.backward()
            optimizer.step()

        # Compute loss on validation set
        model.eval()
        with torch.no_grad():
            if loss_function == 'beta-vae':
                val_nll = 0
                val_kl = 0
                for images, _ in val_dloader:
                    images = images.cuda()
                    q_params, _, recons = model(images)
                    val_nll += ((recons - images)**2).sum()
                    val_kl += models.gaussian_KL(*q_params).sum()
                val_nll = val_nll.sum() / len(val_dset)
                val_kl = val_kl.sum() / len(val_dset)
                val_loss = val_nll + beta_weight * val_kl

                wandb.log({'epoch': epoch, 'val_nll': val_nll.item(),
                          'val_kl': val_kl.item(), 'val_loss': val_loss.item()})
                utils.tprint(f'Validation loss {val_loss.item():.5f}',
                             f'(MSE: {val_nll.item():.5f}, KL: {kl.item():.5f})')
            elif loss_function == 'tcvae':
                raise NotImplementedError("not yet")
            else:
                raise ValueError(f'Loss function {loss_function} is not a valid option.')
        model.train()

        # Check for divergence
        if torch.isnan(val_loss) or torch.isinf(val_loss):
            wandb.run.summary['diverged'] =  True
            wandb.run.summary['best_epoch'] = best_epoch
            wandb.run.summary['best_val_nll'] = best_nll
            wandb.run.finish()
            raise ValueError('Validation loss diverged')

        # Reduce learning rate
        scheduler.step(val_nll)

        # Save best model yet (if needed)
        if val_nll < best_nll:
            utils.tprint('Saving best model')
            best_epoch = epoch
            best_nll = val_nll
            best_model = copy.deepcopy(model).cpu()
            wandb.log({'epoch': epoch, 'model_saved': True})

        # Stop training if validation has not improved in x number of epochs
        if epoch - best_epoch >= stopping_epochs:
            utils.tprint('Stopping training.',
                         f' Validation has not improved in {stopping_epochs} epochs.')
            break

    # Report
    training_time = round((time.time() - start_time) / 60) # in minutes
    utils.tprint(f'Reached max epochs in {training_time} minutes')

    # Record final metrics
    wandb.run.summary['diverged'] =  False
    wandb.run.summary['best_epoch'] = best_epoch
    wandb.run.summary['best_val_nll'] = best_nll

    # Save model
    wandb.save('model.pt')
    torch.save(best_model.state_dict(), path.join(wandb.run.dir, 'model.pt'))

    # Finish wandb session
    wandb.run.finish()

    return best_model
