"""Estimate mutual information using a neural network discriminator."""
import torch
from torch import nn
from torch.nn import functional as F

from dermosxai import utils


class MLP(nn.Module):
    """ A simple two layer MLP. 
    
    Arguments:
        in_features (int): Number of input features.
        num_hidden_features (int): Number of hidden features.
        dropout_rate (float): Amount of input to linear layers that will be dropout. 
            Input dropout is maxed at 0.1.
    """
    def __init__(self, in_features, num_hidden_features=256, dropout_rate=0.25):
        super().__init__()

        # Define MLP
        self.layers = nn.Sequential(
            nn.Dropout(min(dropout_rate, 0.1)),
            nn.Linear(in_features, num_hidden_features, bias=False),
            nn.BatchNorm1d(num_hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            # nn.Linear(num_hidden_features, num_hidden_features // 2),
            # nn.BatchNorm1d(num_hidden_features//2),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout_rate),
            nn.Linear(num_hidden_features, 1),
        )

    def forward(self, x):
        return self.layers(x)

    def init_parameters(self):
        for module in self.layers:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.normal_(module.weight, mean=1, std=0.1)
                nn.init.constant_(module.bias, 0)

class MIEstimator(nn.Module):
    """ Uses a MLP to estimate mutual information between two sets of variables.
    
    This module returns three estimates: Donsker-Varadhan (Belghazi et al. 2018), 
    Jensen-Shannon (Nowozin et al., 2016) and InfoNCE (Oord et al., 2018). See Sec 3.1 in 
    Hjelm et al., 2018 for details of each. They are all black-box methods that rely on 
    the neural estimator to be able to discriminate between samples from the joint p(x,z)
    and samples from the product distribution p(x)p(z). All estimates are lower bounds, 
    i.e., to tighten the MI estimate the neural estimator is optimized to maximize the 
    estimated mutual information.
        
    Arguments:
        in_features (tuple): Tuple with the number of dimensions for each variable.
    
    Returns:
        dv (torch.float): Donsker-Varadhan (MINE) estimate of the MI
        dv_loss (torch.float): Loss to be optimized for the DV MI estimate. See Note.
        jsd (torch.float): The Jensen-Shannon estimate of MI
        infonce (torch.float): The InfoNCE estimate of MI.
        
    Note:
        For the VD estimate, the gradient needs to be corrected to improve perfomance, 
        thus we also return a modified loss to be maximized that will produce the righ 
        gradient. Other MI estimates can be directly optimized
    """
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

        # Create statistics network
        self.stat_network = MLP(sum(in_features))

        # Initialize the exponential moving average of e^T(x, y)
        self.register_buffer('running_exp', torch.tensor(float('nan')))

    def forward(self, x, y):
        """ Returns the MI estimates
        
        Arguments:
            x (torch.Tensor): A (N x num_x_features) tensor with the x features.
            y (torch.Tensor): A (N x num_y_features) tensor with the y features. Each row
                has an x, y sampled from the joint distribution (i.e., paired features).
        """
        # Pass all combinations of x, y through the estimator
        num_samples = x.shape[0]
        xy = torch.cat([x.repeat_interleave(num_samples, dim=0),
                        y.tile(num_samples, 1)], -1)
        stats = self.stat_network(xy).reshape(num_samples, num_samples)

        # Compute DV estimate
        diag = torch.diagonal(stats)
        off_diag = utils.off_diagonal(stats)  # all off-diagonal elements
        logmeanexp = torch.logsumexp(off_diag, 0) - torch.log(torch.tensor(len(off_diag)))
        dv = diag.mean() - logmeanexp

        # Compute dv_loss
        if self.train:
            with torch.no_grad():
                if torch.isnan(self.running_exp):  # very first iteration
                    self.running_exp = torch.exp(logmeanexp)
                else:
                    self.running_exp = 0.9 * self.running_exp + 0.1 * torch.exp(
                        logmeanexp)
        dv_loss = diag.mean() - torch.exp(logmeanexp) / (self.running_exp + 1e-8)

        # Compute JSD
        I = torch.eye(stats.shape[0], device=stats.device)
        sp = F.binary_cross_entropy_with_logits(stats, I, reduction='none')
        jsd = -torch.diagonal(sp).mean() - utils.off_diagonal(sp).mean()

        # Compute InfoNCE
        infonce = torch.diagonal(stats.log_softmax(dim=1)).mean()

        return dv, dv_loss, jsd, infonce

    def init_parameters(self):
        self.stat_network.init_parameters()


def train_mi(train_x, train_y, val_x, val_y, mi_version='mine', batch_size=96, seed=4321,
             learning_rate=0.001, weight_decay=0, decay_epochs=10, lr_decay=-0.1,
             num_epochs=300, stopping_epochs=30, device='cuda'):
    """ Train an MI estimator for the provided data.
    
    Arguments:
        train_x, train_y (torch.tensor): Training sets. Each is a (N x num_features) array, 
            where N should match and num_features can be different between x and y.
        val_x, val_y (np.array): Validation sets. Used for early stopping.
        mi_version (string): Which MI method to use:
            'mine': Optimizes the Donsker-Varadhan bound of the MI as done for MINE.
            'jsd': Optimizes the Jensen-Shannon bound.
            'infonce': Optimizes the noise constrastive estimation bound on the MI.
        batch_size (int): Batch size used for learning the estimator.
        seed (int): Random seed.
        learning_rate (float): Learning rate for the ADAM optimizer.
        weight_decay (float): Wegith decay for the optmizer.
        decay_epochs (int): Numberof epochs to wait before decaying the learning rate if 
            the optimization hasn't improved.
        lr_decay (float): How much to decrease the learning rate when optimization has not 
            improved.
        num_epochs (int): Maximum number of epochs.
        stopping_epochs (int): Stop training after this number of epochs without 
            improvement.
        device (torch.device or str): Where to run the training.
    
    Returns:
        best_model (MIEstimator): Best Mi estimator.
        dvs, dv_losses, jsds, infonces: Training MI estimates. 
        val_dvs, val_dv_losses, val_jsds, val_infonces: MI estimates on validation set.

    Note: 
        This method uses the data as is. Make sure to normalize it before sending it here.
    """
    import copy
    import time

    from torch import optim
    from torch.optim import lr_scheduler
    from torch.utils import data

    # Get datasets
    train_dset = data.TensorDataset(train_x, train_y)
    train_dloader = data.DataLoader(train_dset, batch_size=batch_size, shuffle=True,
                                    num_workers=4, drop_last=True)

    # Set seed
    torch.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    # Get model
    model = MIEstimator((train_x.shape[-1], train_y.shape[-1]))
    model.init_parameters()
    model.train()
    model.to(device)

    # Declare optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=weight_decay)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_decay, mode='max',
                                               patience=decay_epochs)

    # Intialize variables for early stopping
    best_model = copy.deepcopy(model).cpu()
    best_epoch = 0
    best_mi = float('-inf')

    # Logs
    dvs = []
    dv_losses = []
    jsds = []
    infonces = []
    val_dvs = []
    val_dv_losses = []
    val_jsds = []
    val_infonces = []

    # Train
    start_time = time.time()  # in seconds
    for epoch in range(1, num_epochs + 1):
        # Loop over training set
        for x, y in train_dloader:
            # Zero the gradients
            model.zero_grad()

            # Forward
            dv, dv_loss, jsd, infonce = model(x.to(device), y.to(device))
            if mi_version == 'mine':
                loss = -dv_loss
            elif mi_version == 'jsd':
                loss = -jsd
            elif mi_version == 'infonce':
                loss = -infonce
            else:
                raise ValueError(f'MI version {mi_version} not recognized')

            # Log
            dvs.append(dv.item())
            dv_losses.append(dv_loss.item())
            jsds.append(jsd.item())
            infonces.append(infonce.item())

            # Check for divergence
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError('Training loss diverged')

            # Backprop
            loss.backward()
            optimizer.step()

        # Compute loss on validation set
        model.eval()
        with torch.no_grad():
            val_dv, val_dv_loss, val_jsd, val_infonce = model(val_x.to(device),
                                                              val_y.to(device))

            val_dvs.append(val_dv.item())
            val_dv_losses.append(val_dv_loss.item())
            val_jsds.append(val_jsd.item())
            val_infonces.append(val_infonce.item())
        model.train()

        if mi_version == 'mine':
            val_mi = val_dv.item()
        elif mi_version == 'jsd':
            val_mi = val_jsd.item()
        elif mi_version == 'infonce':
            val_mi = val_infonce.item()
        else:
            raise ValueError(f'MI version {mi_version} not recognized')

        # Reduce learning rate
        scheduler.step(val_mi)

        # Save best model yet (if needed)
        if val_mi > best_mi:
            best_epoch = epoch
            best_mi = val_mi
            best_model = copy.deepcopy(model).cpu()

        # Stop training if validation has not improved in x number of epochs
        if epoch - best_epoch >= stopping_epochs:
            utils.tprint('Stopping training.',
                         f' Validation has not improved in {stopping_epochs} epochs.')
            break

    # Report
    training_time = round((time.time() - start_time))  # in minutes
    utils.tprint(f'Reached max epochs in {training_time} seconds')

    return (best_model, dvs, dv_losses, jsds, infonces, val_dvs, val_dv_losses, val_jsds,
            val_infonces)
