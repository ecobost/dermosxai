""" Code to train models with human and learned features and an MI penalty"""
import copy
import itertools
import time
from os import path

import torch
import wandb
from torch import optim
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils import data

from dermosxai import datasets, mi, models, train_abl, transforms, utils


@torch.no_grad()
def get_intermediate_features(model, dset, batch_size=128, device='cuda'):
    """ Gets all human and convnet features from a joint model."""
    model.to(device)
    dloader = data.DataLoader(dset, batch_size=batch_size, num_workers=4)

    human_features = []
    convnet_features = []
    for images, _ in dloader:
        model(images.to(device))
        human_features.append(model.human_features.cpu())
        convnet_features.append(model.convnet_features.cpu())
    human_features = torch.cat(human_features)
    convnet_features = torch.cat(convnet_features)

    return human_features, convnet_features


def make_infinite(dloader):
    """ Infinitely cycle through the dataloader (shuffling the inputs each loop)."""
    while True:
        yield from dloader


def train_joint_with_mi(model, train_dset, val_dset, seed=54321, batch_size=96,
                        learning_rate=0.01, weight_decay=0, mi_learning_rate=1e-3,
                        mi_weight_decay=0, mi_num_epochs_factor=5, mi_lambda=0.1,
                        mi_scaling_epochs=10, num_epochs=200, decay_epochs=4,
                        lr_decay=0.1, stopping_epochs=25, base_lr_factor=1,
                        wandb_group=None, wandb_extra_hyperparams={}):
    """  Train a composite model (concat(human_features, image_features) + linear) for 
    diagnosis with MI penalty.
    
    Uses ADAM optimizer and the validation set for early stopping.
    
    MI estimator is trained in parallel, although it could see more data than the rest of 
    the model, i.e., for each batch (forward plus backward of the model) the MI estimator 
    is trained for mi_num_epochs_factor as many batches. To avoid noisy MI estimates 
    hinder learning at early stages, we also linearly increase the mi penalty from 0 to 
    mi_lambda over mi_scaling_epochs epochs.
        
    Arguments (input):
        model (nn.Module): Pytorch module. Receives an image input, outputs a list of 
            logit vectors to obtain the probabilities for each attribute.
        train_dset (Dataset): Pytorch dataset with training images. Returns a (image, 
            label, attrs) triplet.
        val_dset (Dataset): Validation dataset.
    
    Arguments (MI estimation):
        mi_learning_rate (float): Learning rate for the mutual information estimator.
        mi_weight_decay (float): Regularization strength for the mutual information 
            estimator.
        mi_num_epochs_factor (int): Factor of epochs to train the MI estimator in 
            comparison to the classification model (see docstring above).
    
    Arguments (training):
        seed (int): Random seed for weight initialization and dataloaders.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for ADAM.
        weight_decay (float): Regularization strength.
        mi_lambda (float): Strength for the MI penalty in the loss.
        mi_scaling_epochs (int): MI penalty is linearly scaled from 0 to mi_lambda over 
            this number of epochs.
        num_epochs (int): Number of maximum epochs to train to.
        decay_epochs (int): Number of epochs to wait before decreasing learning rate if 
            validation mcc has not improved.
        lr_decay (float): Factor multiplying learning rate when decaying.
        stopping_epochs (int): Early stop training after this number of epochs without an 
            improvement in validation mcc.
        base_lr_factor (float): How to modify the learning rate to obtain the learning 
            rate for the base feature extractor. Useful for finetuning. Learning rate of 
            base = learning_rate * lr_factor
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
        'weight_decay': weight_decay, 'mi_learning_rate': mi_learning_rate,
        'mi_weight_decay': mi_weight_decay, 'mi_num_epochs_factor': mi_num_epochs_factor,
        'mi_lambda': mi_lambda, 'mi_scaling_epochs': mi_scaling_epochs,
        'num_epochs': num_epochs, 'decay_epochs': decay_epochs, 'lr_decay': lr_decay,
        'stopping_epochs': stopping_epochs, 'base_lr_factor': base_lr_factor,
        **wandb_extra_hyperparams}
    wandb.init(project='dermosxai_joint2', group=wandb_group, config=hyperparams,
               dir='/src/dermosxai/data', tags=['use_dvmi'])

    # Set random seed
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)

    # Create dloaders
    utils.tprint('Creating dloaders')
    train_dloader = data.DataLoader(train_dset, batch_size=batch_size, num_workers=4,
                                    shuffle=True, drop_last=True)
    val_dloader = data.DataLoader(val_dset, batch_size=128, num_workers=4)
    mi_dloader = make_infinite(train_dloader)

    # Set up model
    utils.tprint('Initializing model')
    model.init_parameters()
    model.train()
    model.abl.eval()  # make sure the ABL is always in eval mode
    model.cuda()

    # Set the abl requires_grad to False (optional, to avoid computing those gradients)
    for param in model.abl.parameters():
        param.requires_grad = False

    # Set up MI estimator
    mi_estimator = mi.MIEstimator(
        (sum(model.abl.out_channels), model.extractor.out_channels))
    mi_estimator.init_parameters()
    mi_estimator.train()
    mi_estimator.cuda()

    # Declare optimizer
    utils.tprint('Declaring optimizer')
    base_params = model.extractor.parameters()
    head_params = [*model.linear_human.parameters(), *model.linear_convnet.parameters()]
    optimizer = optim.Adam([{'params': base_params, 'lr': learning_rate * base_lr_factor},
                            {'params': head_params}], lr=learning_rate,
                           weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_decay, mode='max',
                                               patience=decay_epochs, verbose=True)

    # Declare MI optmiizer
    mi_optimizer = optim.Adam(mi_estimator.parameters(), lr=mi_learning_rate,
                              weight_decay=mi_weight_decay)

    # Intialize variables for early stopping
    best_model = copy.deepcopy(model).cpu()
    best_epoch = 0
    best_mcc = float('-inf')
    best_acc = float('-inf')
    best_mi = float('inf')

    # Train
    start_time = time.time()  # in seconds
    for epoch in range(1, num_epochs + 1):
        utils.tprint(f'Epoch {epoch}:')

        # Compute MI for this epoch
        mi_lambda_prop = min((epoch - 1) / mi_scaling_epochs, 1)
        current_mi_lambda = mi_lambda * mi_lambda_prop

        # Record learning rate
        wandb.log({
            'epoch': epoch, 'lr': optimizer.param_groups[1]['lr'],
            'current_mi_lambda': current_mi_lambda})

        # Loop over training set
        for batch_i, (images, labels) in enumerate(train_dloader):
            # Forward
            logits = model(images.cuda())

            # Compute loss
            nll = F.cross_entropy(logits, labels.cuda())
            mi_estimator.eval()
            dv_mi, dv_loss, jsd_mi, infonce_mi = mi_estimator(model.human_features,
                                                              model.convnet_features)
            mi_estimator.train()
            loss = nll + current_mi_lambda * dv_mi

            # Compute other metrics
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1).cpu().numpy()
            accuracy, kappa, mcc, f1, auc, ap = utils.compute_metrics(
                probs, labels.numpy())

            # Log loss
            wandb.log({
                'epoch': epoch, 'batch': batch_i, 'nll': nll.item(),
                'dv_mi': dv_mi.item(), 'dv_loss': dv_loss.item(), 'jsd_mi': jsd_mi.item(),
                'infonce_mi': infonce_mi.item(), 'loss': loss.item(),
                'accuracy': accuracy, 'kappa': kappa, 'mcc': mcc, 'f1': f1, 'auc': auc,
                'ap': ap})
            if batch_i % 10 == 0:
                utils.tprint(f'Training loss {loss.item():.3f} (Acc: {accuracy:.3f}, ',
                             f'MCC: {mcc:.3f})  MI: {dv_mi:.3f}')

            # Check for divergence
            if torch.isnan(loss) or torch.isinf(loss):
                wandb.run.summary['diverged'] = True
                wandb.run.summary['best_epoch'] = best_epoch
                wandb.run.summary['best_val_mcc'] = best_mcc
                wandb.run.summary['best_val_acc'] = best_acc
                wandb.run.summary['best_val_mi'] = best_mi
                wandb.run.finish()
                raise ValueError('Training loss diverged')

            # Backprop
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Backprop for the MI estimator (mi_num_epochs_factor times)
            model.eval()
            for images, _ in itertools.islice(mi_dloader, mi_num_epochs_factor):
                # Forward
                model(images.cuda())

                # Backprop for the MI estimator
                mi_estimator.zero_grad()
                dv_loss = mi_estimator(model.human_features.detach(),
                                       model.convnet_features.detach())[0]
                (-dv_loss).backward()
                mi_optimizer.step()
            model.train()
            model.abl.eval()

        # Compute loss on validation set
        model.eval()
        mi_estimator.eval()
        with torch.no_grad():
            val_logits = []
            val_labels = []
            val_human_features = []
            val_convnet_features = []
            for images, labels in val_dloader:
                val_logits.append(model(images.cuda()).cpu())
                val_human_features.append(model.human_features.cpu())
                val_convnet_features.append(model.convnet_features.cpu())
                val_labels.append(labels.cpu())
            val_logits = torch.cat(val_logits)
            val_human_features = torch.cat(val_human_features)
            val_convnet_features = torch.cat(val_convnet_features)
            val_labels = torch.cat(val_labels)

            # Compute loss
            val_nll = F.cross_entropy(val_logits, val_labels)
            mi_estimator.cpu()
            val_dv_mi, val_dv_loss, val_jsd_mi, val_infonce_mi = mi_estimator(
                val_human_features, val_convnet_features)
            mi_estimator.cuda()
            val_loss = val_nll + current_mi_lambda * val_dv_mi

            # Compute metrics
            val_probs = F.softmax(val_logits, dim=-1)
            val_accuracy, val_kappa, val_mcc, val_f1, val_auc, val_ap = utils.compute_metrics(
                val_probs.numpy(), val_labels.numpy())

            # Log
            wandb.log({
                'epoch': epoch, 'val_nll': val_nll.item(), 'val_dv_mi': val_dv_mi.item(),
                'val_dv_loss': val_dv_loss.item(), 'val_jsd_mi': val_jsd_mi.item(),
                'val_infonce_mi': val_infonce_mi.item(), 'val_loss': val_loss.item(),
                'val_accuracy': val_accuracy, 'val_kappa': val_kappa, 'val_mcc': val_mcc,
                'val_f1': val_f1, 'val_auc': val_auc, 'val_ap': val_ap})
            utils.tprint(
                f'Validation loss {val_loss.item():.3f} (Acc: {val_accuracy:.3f}, ',
                f'MCC: {val_mcc:.3f}) MI: {val_dv_mi:.3f}')
        model.train()
        model.abl.eval()
        mi_estimator.train()

        # Check for divergence
        if torch.isnan(val_loss) or torch.isinf(val_loss):
            wandb.run.summary['diverged'] = True
            wandb.run.summary['best_epoch'] = best_epoch
            wandb.run.summary['best_val_mcc'] = best_mcc
            wandb.run.summary['best_val_acc'] = best_acc
            wandb.run.summary['best_val_mi'] = best_mi
            wandb.run.finish()
            raise ValueError('Validation loss diverged')
        if epoch > mi_scaling_epochs:
            scheduler.step(val_mcc)

        # Save best model yet (if needed)
        if val_mcc > best_mcc and epoch > mi_scaling_epochs:
            utils.tprint(f'Saving best model. Improvement: {val_mcc-best_mcc:.3f}')
            best_epoch = epoch
            best_mcc = val_mcc
            best_acc = val_accuracy
            best_mi = val_dv_mi.item()
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
    wandb.run.summary['best_val_mi'] = best_mi

    # Save model
    wandb.save('model.pt')
    torch.save(best_model.state_dict(), path.join(wandb.run.dir, 'model.pt'))

    # Recompute MI estimates with a estimator trained from scratch
    utils.tprint('Training the final MI estimator.')
    best_model.cuda()
    best_model.eval()
    with torch.no_grad():
        train_transform = train_dset.transform
        train_dset.transform = val_dset.transform  # do not augment
        train_human_features, train_convnet_features = get_intermediate_features(
            best_model, train_dset)
        val_human_features, val_convnet_features = get_intermediate_features(
            best_model, val_dset)
        train_dset.transform = train_transform
    best_model.train()
    best_model.abl.eval()
    mi_estimator = mi.train_mi(train_human_features, train_convnet_features,
                               val_human_features, val_convnet_features)[0]
    mi_estimator.eval()
    with torch.no_grad():
        val_dv_mi, _, val_jsd_mi, val_infonce_mi = mi_estimator(
            val_human_features, val_convnet_features)
    wandb.run.summary['final_val_dv_mi'] = val_dv_mi
    wandb.run.summary['final_val_jsd_mi'] = val_jsd_mi
    wandb.run.summary['final_val_infonce_mi'] = val_infonce_mi

    # Save estimator
    wandb.save('estimator.pt')
    torch.save(mi_estimator.state_dict(), path.join(wandb.run.dir, 'estimator.pt'))

    # Finish wandb session
    wandb.run.finish()

    return best_model


def train_HAM10000():
    # Get dsets
    utils.tprint('Getting datasets')
    train_dset = datasets.HAM10000('train')
    val_dset = datasets.HAM10000('val')

    # Add transforms
    train_transform, val_transform = transforms.get_HAM10000_transforms(
        train_dset.img_mean, train_dset.img_std)
    train_dset.transform = train_transform
    val_dset.transform = val_transform

    # Get attribute predictor
    utils.tprint('Setting up model...')
    abl_model = train_abl.get_HAM10000_AbL()

    # Get feature extractor
    extractor = models.ResNetBase(num_blocks=3)

    # Create joint model
    num_classes = train_dset.labels.max() + 1
    model = models.JointWithLinearHead(abl_model, extractor, out_channels=num_classes)

    # Train
    for seed in [54321, 9845, 988, 1701, 5318]:
        for learning_rate, base_lr_factor in [(1e-4, 1), (1e-3, 1e-1), (1e-2, 1e-2)]:
            for mi_lambda in [0, 0.1, 0.33, 0.66, 1, 3.33, 6.66, 10]:
                try:
                    train_joint_with_mi(copy.deepcopy(model), train_dset, val_dset,
                                        learning_rate=learning_rate,
                                        base_lr_factor=base_lr_factor,
                                        mi_lambda=mi_lambda, seed=seed,
                                        wandb_group='ham10000',
                                        wandb_extra_hyperparams={'base': 'resnet'})
                except ValueError:  # ignore convergence error
                    pass

    # # Get feature extractor
    # extractor = models.ConvNetBase()

    # # Create joint model
    # num_classes = train_dset.labels.max() + 1
    # model = models.JointWithLinearHead(abl_model, extractor, out_channels=num_classes)

    # # Train
    # for learning_rate in [1e-4, 1e-3, 1e-2]:
    #     for mi_lambda in [0, 0.1, 0.33, 0.66, 1, 3.33, 6.66, 10]:
    #         try:
    #             train_joint_with_mi(copy.deepcopy(model), train_dset, val_dset,
    #                                 learning_rate=learning_rate,
    #                                 mi_lambda=mi_lambda,
    #                                 wandb_group='ham10000',
    #                                 wandb_extra_hyperparams={'base': 'convnet'})
    #         except ValueError:  # ignore convergence error
    #             pass



def get_HAM10000_joint(wandb_path):
    """ Downloads pretrained weights from wandb and loads model.
    
    Arguments:
        wandb_path (str): Name of the run as recorded by wandb usually something like 
            "username/project_name/run_id".
                   
    Returns:
        model (nn.Module): A JointWithLinearHead model with the pretrained weights.
        mi_estimator (nn.Module): The pretrained MIEstimator for human and convnet 
            features outputted by the model.
    """
    num_classes = 4  # datasets.HAM10000('train').labels.max() + 1
    abl_model = train_abl.get_HAM10000_AbL()
    extractor = models.ResNetBase(num_blocks=3)
    model = models.JointWithLinearHead(abl_model, extractor, out_channels=num_classes)

    # Get model from wandb
    model_path = wandb.restore('model.pt', run_path=wandb_path, replace=True,
                               root='/tmp').name  # downloads weights
    model.load_state_dict(torch.load(model_path))

    # Get MI estimator too
    mi_estimator = mi.MIEstimator(
        (sum(model.abl.out_channels), model.extractor.out_channels))
    estimator_path = wandb.restore('estimator.pt', run_path=wandb_path, replace=True,
                                   root='/tmp').name
    mi_estimator.load_state_dict(torch.load(estimator_path))

    return model, mi_estimator


def evaluate_HAM10000(wandb_path):
    """ Computes the evaluaton metrics (and MI) in the test set.
    
    Arguments:
        wandb_path (str): Name of the run as recorded by wandb usually something like 
            "username/project_name/run_id".
            
    Returns:
        metrics (tuple): Six metrics as returned by utils.compute_metrics 
            (accuracy, kappa, mcc, f1, auc, ap).
        mi_estimates (tuple): MI estimates as returned by the MI estimator
            (dv, dv_loss, jsd, infonce).
    """
    # Get data
    train_dset = datasets.HAM10000('train')
    test_dset = datasets.HAM10000('test')

    # Get transforms
    _, test_transform = transforms.get_HAM10000_transforms(train_dset.img_mean,
                                                           train_dset.img_std)
    test_dset.transform = test_transform

    # Get dloader
    dloader = data.DataLoader(test_dset, batch_size=128, num_workers=4)

    # Get model
    model, mi_estimator = get_HAM10000_joint(wandb_path)
    model.cuda()
    model.eval()
    #mi_estimator.cuda()
    mi_estimator.eval()

    # Pass through model
    logits = []
    labels = []
    human_features = []
    convnet_features = []
    with torch.no_grad():
        for images, labels_ in dloader:
            logits.append(model(images.cuda()).cpu())
            human_features.append(model.human_features.cpu())
            convnet_features.append(model.convnet_features.cpu())
            labels.append(labels_.cpu())
    logits = torch.cat(logits)
    human_features = torch.cat(human_features)
    convnet_features = torch.cat(convnet_features)
    labels = torch.cat(labels)

    # Compute metrics
    probs = F.softmax(logits, dim=-1).numpy()
    metrics = utils.compute_metrics(probs, labels.numpy())
    print('Metrics (HAM10000)', metrics)

    # Compute MI
    with torch.no_grad():
        mi_estimates = mi_estimator(human_features, convnet_features)
    print('MI estimates (HAM10000);', mi_estimates)

    return metrics, mi_estimates



def train_DDSM():
    # Get dsets
    utils.tprint('Getting datasets')
    train_dset = datasets.DDSM('train')
    val_dset = datasets.DDSM('val')

    # Add transforms
    train_transform, val_transform = transforms.get_DDSM_transforms(
        train_dset.img_mean, train_dset.img_std, make_rgb=True)
    train_dset.transform = train_transform
    val_dset.transform = val_transform

    # Get attribute predictor
    utils.tprint('Setting up model...')
    abl_model = train_abl.get_DDSM_AbL()

    # Get feature extractor
    extractor = models.ResNetBase(num_blocks=3)

    # Create joint model
    num_classes = train_dset.labels.max() + 1
    model = models.JointWithLinearHead(abl_model, extractor, out_channels=num_classes)

    # Train
    for seed in [54321, 9845, 988, 1701, 5318]:
        for learning_rate, base_lr_factor in [(1e-4, 1), (1e-3, 1e-1), (1e-2, 1e-2)]:
            for mi_lambda in [0, 0.1, 0.33, 0.66, 1, 3.33, 6.66, 10]:
                try:
                    train_joint_with_mi(copy.deepcopy(model), train_dset, val_dset,
                                        learning_rate=learning_rate,
                                        base_lr_factor=base_lr_factor, mi_lambda=mi_lambda,
                                        wandb_group='ddsm', seed=seed,
                                        wandb_extra_hyperparams={'base': 'resnet'})
                except ValueError:  # ignore convergence error
                    pass

    # # Get feature extractor
    # extractor = models.ConvNetBase()

    # # Create joint model
    # num_classes = train_dset.labels.max() + 1
    # model = models.JointWithLinearHead(abl_model, extractor, out_channels=num_classes)

    # # Train
    # for learning_rate in [1e-4, 1e-3, 1e-2]:
    #     for weight_decay in [0, 1e-2]:
    #         for mi_lambda in [0, 1e-1, 1e0, 1e1]:
    #             try:
    #                 train_joint_with_mi(copy.deepcopy(model), train_dset, val_dset,
    #                                     learning_rate=learning_rate,
    #                                     weight_decay=weight_decay, mi_lambda=mi_lambda,
    #                                     wandb_group='ddsm',
    #                                     wandb_extra_hyperparams={'base': 'convnet'})
    #             except ValueError:  # ignore convergence error
    #                 pass


def get_DDSM_joint(wandb_path):
    """ Downloads pretrained weights from wandb and loads model.
    
    Arguments:
        wandb_path (str): Name of the run as recorded by wandb usually something like 
            "username/project_name/run_id".
                   
    Returns:
        model (nn.Module): A JointWithLinearHead model with the pretrained weights.
        mi_estimator (nn.Module): The pretrained MIEstimator for human and convnet 
            features outputted by the model.
    """
    num_classes = 2  # datasets.DDSM('train').labels.max() + 1
    abl_model = train_abl.get_DDSM_AbL()
    extractor = models.ResNetBase(num_blocks=3)
    model = models.JointWithLinearHead(abl_model, extractor, out_channels=num_classes)

    # Get model from wandb
    model_path = wandb.restore('model.pt', run_path=wandb_path, replace=True,
                               root='/tmp').name  # downloads weights
    model.load_state_dict(torch.load(model_path))

    # Get MI estimator too
    mi_estimator = mi.MIEstimator(
        (sum(model.abl.out_channels), model.extractor.out_channels))
    estimator_path = wandb.restore('estimator.pt', run_path=wandb_path, replace=True,
                                   root='/tmp').name
    mi_estimator.load_state_dict(torch.load(estimator_path))

    return model, mi_estimator


def evaluate_DDSM(wandb_path):
    """ Computes the evaluaton metrics (and MI) in the test set.
    
    Arguments:
        wandb_path (str): Name of the run as recorded by wandb usually something like 
            "username/project_name/run_id".
            
    Returns:
        metrics (tuple): Six metrics as returned by utils.compute_metrics 
            (accuracy, kappa, mcc, f1, auc, ap).
        mi_estimates (tuple): MI estimates as returned by the MI estimator
            (dv, dv_loss, jsd, infonce).
    """
    # Get data
    train_dset = datasets.DDSM('train')
    test_dset = datasets.DDSM('test')

    # Get transforms
    _, test_transform = transforms.get_DDSM_transforms(train_dset.img_mean,
                                                           train_dset.img_std, make_rgb=True)
    test_dset.transform = test_transform

    # Get dloader
    dloader = data.DataLoader(test_dset, batch_size=128, num_workers=4)

    # Get model
    model, mi_estimator = get_DDSM_joint(wandb_path)
    model.cuda()
    model.eval()
    #mi_estimator.cuda()
    mi_estimator.eval()

    # Pass through model
    logits = []
    labels = []
    human_features = []
    convnet_features = []
    with torch.no_grad():
        for images, labels_ in dloader:
            logits.append(model(images.cuda()).cpu())
            human_features.append(model.human_features.cpu())
            convnet_features.append(model.convnet_features.cpu())
            labels.append(labels_.cpu())
    logits = torch.cat(logits)
    human_features = torch.cat(human_features)
    convnet_features = torch.cat(convnet_features)
    labels = torch.cat(labels)

    # Compute metrics
    probs = F.softmax(logits, dim=-1).numpy()
    metrics = utils.compute_metrics(probs, labels.numpy())
    print('Metrics (DDSM)', metrics)

    # Compute MI
    with torch.no_grad():
        mi_estimates = mi_estimator(human_features, convnet_features)
    print('MI estimates (DDSM);', mi_estimates)

    return metrics, mi_estimates