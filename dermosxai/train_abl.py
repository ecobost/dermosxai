"""Training code for the attribute prediction network. """
import torch
from torch.utils import data
from torch import optim
from torch.optim import lr_scheduler
from torch.nn import functional as F
import copy
import time
import wandb
from os import path

from dermosxai import utils
from dermosxai import datasets
from dermosxai import transforms
from dermosxai import models


def finetune(model, train_dset, val_dset, seed=1, batch_size=64, learning_rate=0.01,
             weight_decay=1e-5, num_epochs=200, decay_epochs=5, lr_decay=0.01,
             stopping_epochs=20, base_lr_factor=0.01, wandb_group=None,
             wandb_extra_hyperparams={}):
    """  Train attribute prediction model with different learning rate for base and head.
    
    Uses ADAM optimizer and the validation set for early stopping.
    
    Arguments (input):
        model (nn.Module): Pytorch module. Receives an image input, outputs a list of 
            logit vectors to obtain the probabilities for each attribute.
        train_dset (Dataset): Pytorch dataset with training images. Returns a (image, 
            label, attrs) triplet.
        val_dset (Dataset): Validation dataset.
    
    Arguments (training):
        seed (int): Random seed for weight initialization and dataloaders.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for ADAM.
        weight_decay (float): Regularization strength.
        num_epochs (int): Number of maximum epochs to train to.
        decay_epochs (int): Number of epochs to wait before decreasing learning rate if 
            validation mcc has not improved.
        lr_decay (float): Factor multiplying learning rate when decaying.
        stopping_epochs (int): Early stop training after this number of epochs without an 
            improvement in validation mcc.
        base_lr_factor (float): How to modify the learning rate to obtain the learning 
            rate for the base. Learning rate of base = learning_rate * lr_factor
        wandb_group (string): Group send to wandb during init. Helps group different runs, 
            e.g., different Resnet architectures or different datases and so on.
        wandb_extra_hyperparams (dict): Dictionary with any extra hyperparameters that 
            need to be saved in wandb.            
    
    Return:
        best_model (nn.Module): Trained model
    """
    # Save hyperparams in wandb
    hyperparams = {
        'seed': seed, 'batch_size': batch_size, 'learning_rate': learning_rate,
        'weight_decay': weight_decay, 'num_epochs': num_epochs,
        'decay_epochs': decay_epochs, 'lr_decay': lr_decay,
        'stopping_epochs': stopping_epochs, 'base_lr_factor': base_lr_factor,
        **wandb_extra_hyperparams}
    wandb.init(project='dermosxai_abl', group=wandb_group, config=hyperparams,
               dir='/src/dermosxai/data')

    # Set random seed
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)

    # Create dloaders
    utils.tprint('Creating dloaders')
    train_dloader = data.DataLoader(
        train_dset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
    )
    val_dloader = data.DataLoader(val_dset, batch_size=128, num_workers=4)

    # Set up model
    utils.tprint('Initializing model')
    model.init_parameters()
    model.train()
    model.cuda()

    # Declare optimizer
    utils.tprint('Declaring optimizer')
    base_lr = learning_rate * base_lr_factor
    optimizer = optim.Adam([{'params': model.base.parameters(), 'lr': base_lr}, {
        'params': model.head.parameters()}], lr=learning_rate, weight_decay=weight_decay)
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
        wandb.log({'epoch': epoch, 'lr': optimizer.param_groups[1]['lr']})

        # Loop over training set
        for batch_i, (images, _, attrs) in enumerate(train_dloader):
            # Zero the gradients
            model.zero_grad()

            # Forward
            logits_per_attr = model(images.cuda())  # list of logit vectors

            # Compute average loss (and metrics) across attributes
            xents = []
            metrics = []
            for lgt, lbl in zip(logits_per_attr, attrs.T):
                xents.append(F.cross_entropy(lgt, lbl.cuda()))

                with torch.no_grad():
                    probs = F.softmax(lgt.detach(), dim=-1).cpu().numpy()
                metrics.append(utils.compute_metrics(probs, lbl.numpy()))
            loss = sum(xents) / len(xents)
            accuracy, kappa, mcc, f1, auc, ap = [sum(m) / len(m) for m in zip(*metrics)]

            # Log loss
            wandb.log({
                'epoch': epoch, 'batch': batch_i, 'loss': loss.item(),
                'accuracy': accuracy, 'kappa': kappa, 'mcc': mcc, 'f1': f1, 'auc': auc,
                'ap': ap})
            if batch_i % 10 == 0:
                utils.tprint(f'Training loss {loss.item():.3f} (Acc: {accuracy:.3f}, ',
                             f'MCC: {mcc:.3f})')

            # Check for divergence
            if torch.isnan(loss) or torch.isinf(loss):
                wandb.run.summary['diverged'] = True
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
            for images, _, attrs in val_dloader:
                val_logits.append([lgt.detach().cpu() for lgt in model(images.cuda())])
                val_labels.append(attrs.detach().cpu())
            val_logits_per_attr = [torch.cat(lgts) for lgts in zip(*val_logits)]
            val_labels = torch.cat(val_labels)

            # Compute average loss (and metrics) across attributes
            val_xents = []
            val_metrics = []
            for lgt, lbl in zip(val_logits_per_attr, val_labels.T):
                val_xents.append(F.cross_entropy(lgt, lbl))

                with torch.no_grad():
                    probs = F.softmax(lgt.detach(), dim=-1).cpu().numpy()
                val_metrics.append(utils.compute_metrics(probs, lbl.numpy()))
            val_loss = sum(val_xents) / len(val_xents)
            val_accuracy, val_kappa, val_mcc, val_f1, val_auc, val_ap = [
                sum(m) / len(m) for m in zip(*metrics)]

            # Log
            wandb.log({
                'epoch': epoch, 'val_loss': val_loss.item(), 'val_accuracy': val_accuracy,
                'val_kappa': val_kappa, 'val_mcc': val_mcc, 'val_f1': val_f1,
                'val_auc': val_auc, 'val_ap': val_ap})
            utils.tprint(
                f'Validation loss {val_loss.item():.3f} (Acc: {val_accuracy:.3f}, ',
                f'MCC: {val_mcc:.3f})')
        model.train()

        # Check for divergence
        if torch.isnan(val_loss) or torch.isinf(val_loss):
            wandb.run.summary['diverged'] = True
            wandb.run.summary['best_epoch'] = best_epoch
            wandb.run.summary['best_val_mcc'] = best_mcc
            wandb.run.summary['best_val_acc'] = best_acc
            wandb.run.finish()
            raise ('Validation loss diverged')
        scheduler.step(val_mcc)

        # Save best model yet (if needed)
        if val_mcc > best_mcc:
            utils.tprint(f'Saving best model. Improvement: {val_mcc-best_mcc:.3f} ')
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
    training_time = round((time.time() - start_time) / 60)  # in minutes
    utils.tprint(f'Reached max epochs in {training_time} minutes')

    # Record final metrics
    wandb.run.summary['diverged'] = False
    wandb.run.summary['best_epoch'] = best_epoch
    wandb.run.summary['best_val_mcc'] = best_mcc
    wandb.run.summary['best_val_acc'] = best_acc

    # Save model
    wandb.save('model.pt')
    torch.save(best_model.state_dict(), path.join(wandb.run.dir, 'model.pt'))

    # Finish wandb session
    wandb.run.finish()

    return best_model


def train_DDSM_resnets():
    # Get dsets
    train_dset = datasets.DDSM('train', return_attributes=True)
    val_dset = datasets.DDSM('val', return_attributes=True)

    # Add transforms
    train_transform, val_transform = transforms.get_DDSM_transforms(
        train_dset.img_mean, train_dset.img_std, make_rgb=True)
    train_dset.transform = train_transform
    val_dset.transform = val_transform

    # Compute num_values_per_attr
    num_values_per_attr = train_dset.attributes.max(0) + 1

    # Set hyperparams
    for resnet_block in range(1, 5):
        for learning_rate in [1e-4, 1e-3, 1e-2, 1e-1]:
            for base_lr_factor in [0, 1e-4, 1e-3, 1e-2]:
                for weight_decay in [0, 1e-5, 1e-3, 1e-1]:
                    # Define model
                    model = models.ResNetPlusMultiLinear(num_blocks=resnet_block,
                                                         out_channels=num_values_per_attr)

                    # Train
                    finetune(
                        model, train_dset, val_dset, learning_rate=learning_rate,
                        weight_decay=weight_decay, base_lr_factor=base_lr_factor,
                        wandb_group='ddsm', wandb_extra_hyperparams={
                            'base': 'resnet', 'resnet_block': resnet_block})
    """
    Selected hyperparams: resnet_block 3, lr 0.01, base_lr_factor 0.01, weight_decay 1e-5
    MCC: 1.0 for training and validation.
    Name of wandb run: glorious-serenity-174
    """

# ended up not training these, resnet achieves 100% train/val accuracy
# def train_DDSM_convnet():
#     # Get dsets
#     train_dset = datasets.DDSM('train', return_attributes=True)
#     val_dset = datasets.DDSM('val', return_attributes=True)

#     # Add transforms
#     train_transform, val_transform = transforms.get_DDSM_transforms(
#         train_dset.img_mean, train_dset.img_std)
#     train_dset.transform = train_transform
#     val_dset.transform = val_transform

#     # Compute num_values_per_attr
#     num_values_per_attr = train_dset.attributes.max(0) + 1

#     # Set hyperparams
#     for learning_rate in [1e-4, 1e-3, 1e-2, 1e-1]:
#         for weight_decay in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
#             # Define model
#             model = models.ConvNetPlustMultiLinear(in_channels=1,
#                                                    out_channels=num_values_per_attr)

#             # Train
#             finetune(model, train_dset, val_dset, learning_rate=learning_rate,
#                      weight_decay=weight_decay, wandb_group='ddsm',
#                      wandb_extra_hyperparams={'base': 'convnet'})


def train_IAD_resnets():
    """
    Note: I use 85% of the data for training and 15% as validation. I do not report 
    results in this dataset so I don't need a test set.
    """
    # Get dsets
    train_dset = datasets.IAD('train', return_attributes=True,
                              split_proportion=(0.85, 0.15))
    val_dset = datasets.IAD('val', return_attributes=True, split_proportion=(0.85, 0.15))

    # Add transforms
    use_full_augmentations = False
    train_transform, val_transform = transforms.get_IAD_transforms(
        train_dset.img_mean, train_dset.img_std,
        use_full_augmentations=use_full_augmentations)
    train_dset.transform = train_transform
    val_dset.transform = val_transform

    # Compute num_values_per_attr
    num_values_per_attr = train_dset.attributes.max(0) + 1

    # Set hyperparams
    for resnet_block in range(1, 5):
        for learning_rate in [1e-3, 1e-2, 1e-1, 1e0]:
            for base_lr_factor in [0, 1e-4, 1e-3, 1e-2]:
                for weight_decay in [0, 1e-5, 1e-3, 1e-1]:
                    # Define model
                    model = models.ResNetPlusMultiLinear(num_blocks=resnet_block,
                                                         out_channels=num_values_per_attr)

                    # Train
                    finetune(
                        model, train_dset, val_dset, learning_rate=learning_rate,
                        weight_decay=weight_decay, base_lr_factor=base_lr_factor,
                        wandb_group='iad', wandb_extra_hyperparams={
                            'base': 'resnet', 'resnet_block': resnet_block,
                            'full_augmentation': use_full_augmentations})

# ended up  not training these
# def train_IAD_convnet():
#     """
#     Note: I use 85% of the data for training and 15% as validation. I do not report
#     results in this dataset so I don't need a test set.
#     """
#     # Get dsets
#     train_dset = datasets.IAD('train', return_attributes=True,
#                               split_proportion=(0.85, 0.15))
#     val_dset = datasets.IAD('val', return_attributes=True, split_proportion=(0.85, 0.15))

#     # Add transforms
#     use_full_augmentations = False
#     train_transform, val_transform = transforms.get_IAD_transforms(
#         train_dset.img_mean, train_dset.img_std,
#         use_full_augmentations=use_full_augmentations)
#     train_dset.transform = train_transform
#     val_dset.transform = val_transform

#     # Compute num_values_per_attr
#     num_values_per_attr = train_dset.attributes.max(0) + 1

#     # Set hyperparams
#     for learning_rate in [1e-4, 1e-3, 1e-2, 1e-1]:
#         for weight_decay in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
#             # Define model
#             model = models.ConvNetPlustMultiLinear(in_channels=3,
#                                                    out_channels=num_values_per_attr)

#             # Train
#             finetune(
#                 model, train_dset, val_dset, learning_rate=learning_rate,
#                 weight_decay=weight_decay, wandb_group='iad', wandb_extra_hyperparams={
#                     'base': 'convnet', 'full_augmentation': use_full_augmentations})
