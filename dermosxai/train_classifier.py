""" Code to train HAM classification models. """
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
    # test some initial hyperparams
    # import itertools
    # for lr, wd, a in itertools.product([1e-4, 1e-3, 1e-2], [0, 1e-4, 1e-2], ['flips', 'full']) :
    #     yield {'learning_rate': lr, 'weight_decay': wd, 'augmentation': a}

    #return [{'weight_decay': 1e-6}, {'learning_rate': 3e-3}, {'batch_size': 128}]
    return []


def train_all():
    for params in get_hyperparams():
        try:
            train(**params)
        except ValueError:
            pass


def train_one(param_id):
    """ Utility function to run one hyperparam. Used to coordinate diff runs in the 
    cluster.
    
    Arguments:
        param_id: Which hyperparam set to train
    """
    all_params = list(get_hyperparams())
    selected_params = all_params[param_id]
    return train(**selected_params)


def train(seed=19, batch_size=64, learning_rate=0.001, weight_decay=0,
          num_epochs=200, decay_epochs=5, lr_decay=0.1, stopping_epochs=25,
          augmentation='flips'):
    """ Trains a classifier with ADAM and early stopping.
    
    Arguments:
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
        augmentation (str): How to augment the training images. One of 'flips' or 'full'.
            'flips': Horizontal and vertical flips.
            'full': Add an affine transform (rotations, translations, shear and scale) and 
                color jittering (contrast, brightness, saturations and hue).
    
    Returns: 
        model (nn.Module): The trained pytorch module.
    """
    # Log wandb hyperparams
    hyperparams = {
        'seed': seed, 'batch_size': batch_size,
        'learning_rate': learning_rate, 'weight_decay': weight_decay,
        'num_epochs': num_epochs, 'decay_epochs': decay_epochs, 'lr_decay': lr_decay,
        'stopping_epochs': stopping_epochs, 'augmentation': augmentation}
    wandb.init(project='dermosxai_classifier', group='ham10K-only', config=hyperparams,
               dir='/src/dermosxai/data', tags=['encoder'])

    # Set random seed
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)

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
    model = models.ConvNet(out_channels=4)
    model.init_parameters()
    model.train()
    model.cuda()

    # Declare optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_decay, mode='max',
                                               patience=decay_epochs, verbose=True)

    # Intialize variables for early stopping
    best_model = copy.deepcopy(model).cpu()
    best_epoch = 0
    best_mcc = float('-inf')
    best_acc = float('-inf')

    # Train
    start_time = time.time()  # in seconds
    for epoch in range(1, num_epochs + 1):
        utils.tprint(f'Epoch {epoch}:')

        # Record learning rate
        wandb.log({'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']})

        # Loop over training set
        for batch_i, (images, labels) in enumerate(train_dloader):
            # Zero the gradients
            model.zero_grad()

            # Forward
            logits = model(images.cuda())

            # Compute loss
            loss = F.cross_entropy(logits, labels.cuda())

            # Compute other metrics
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
            accuracy, kappa, mcc, f1, auc, ap = utils.compute_metrics(probs,
                                                                      labels.numpy())

            # Log loss
            wandb.log({
                'epoch': epoch,
                'batch': batch_i,
                'loss': loss.item(),
                'accuracy': accuracy,
                'kappa': kappa,
                'mcc': mcc,
                'f1': f1,
                'auc': auc,
                'ap': ap})
            if batch_i % 10 == 0:
                utils.tprint(f'Training loss {loss.item():.3f} (Acc: {accuracy:.3f}, ',
                             f'MCC: {mcc:.3f})')

            # Check for divergence
            if torch.isnan(loss) or torch.isinf(loss):
                wandb.run.summary['diverged'] =  True
                wandb.run.summary['best_epoch'] = best_epoch
                wandb.run.summary['best_val_mcc'] = best_mcc
                wandb.run.summary['best_val_acc'] = best_acc
                wandb.run.finish()
                raise ValueError('Training loss diverged')

            # Backprop
            loss.backward()
            optimizer.step()

        # Compute loss on validation set
        model.eval()
        with torch.no_grad():
            val_logits = []
            val_labels = []
            for images, labels in val_dloader:
                val_logits.append(model(images.cuda()).detach().cpu())
                val_labels.append(labels.detach().cpu())
            val_logits = torch.cat(val_logits)
            val_labels = torch.cat(val_labels)

            # Compute loss
            val_loss = F.cross_entropy(val_logits, val_labels)

            # Compute metrics
            val_probs = F.softmax(val_logits, dim=-1).detach().cpu().numpy()
            val_accuracy, val_kappa, val_mcc, val_f1, val_auc, val_ap = utils.compute_metrics(val_probs,
                                                                      val_labels.numpy())

            wandb.log({
                'epoch': epoch,
                'val_loss': val_loss.item(),
                'val_accuracy': val_accuracy,
                'val_kappa': val_kappa,
                'val_mcc': val_mcc,
                'val_f1': val_f1,
                'val_auc': val_auc,
                'val_ap': val_ap})
            utils.tprint(
                f'Validation loss {val_loss.item():.3f} (Acc: {val_accuracy:.3f}, ',
                             f'MCC: {val_mcc:.3f})')
        model.train()

        # Check for divergence
        if torch.isnan(val_loss) or torch.isinf(val_loss):
            wandb.run.summary['diverged'] =  True
            wandb.run.summary['best_epoch'] = best_epoch
            wandb.run.summary['best_val_mcc'] = best_mcc
            wandb.run.summary['best_val_acc'] = best_acc
            wandb.run.finish()
            raise ValueError('Validation loss diverged')
        scheduler.step(val_mcc)

        # Save best model yet (if needed)
        if val_mcc > best_mcc:
            utils.tprint('Saving best model')
            best_epoch = epoch
            best_mcc = val_mcc
            best_acc = val_accuracy
            best_model = copy.deepcopy(model).cpu()

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
    wandb.run.summary['best_val_mcc'] = best_mcc
    wandb.run.summary['best_val_acc'] = best_acc

    # Save model
    wandb.save('model.pt')
    torch.save(best_model.state_dict(), path.join(wandb.run.dir, 'model.pt'))

    # Finish wandb session
    wandb.run.finish()

    return best_model
