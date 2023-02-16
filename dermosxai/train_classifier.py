""" Code to train diagnosis classification models. """
from os import path

import h5py
import numpy as np

from dermosxai import datasets, models, train_abl, transforms, utils


def train_linear(X, y, wds=np.logspace(-4, 4, 17)):
    """ Train linear models (sklearn.linear_model).
    
    Binary classification uses a sigmoid activation function to produce final probabilities 
    for the positive class, while multi-class classification (>2)uses a softmax. Thus, for 
    binary classification the model willl only learn weights for one class.
    
    Arguments:
        X (np.array): Datasets (num_examples x num_features)
        y, val_y (np array): Targets (num_examples) as categorical variables (0-n_classes).
        wds (list of floats): Regularization strengths to try.
    
    Returns:
        weights (np.array): A num_regs x num_classes x num_features array with the weights
            for each regularization weight.
        biases (np.array): A num_regs x num_classes array with the biases.
    """
    import warnings

    from sklearn import exceptions, linear_model

    weights = []
    biases = []
    for reg in wds:
        model = linear_model.LogisticRegression(C=1/reg, max_iter=500)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", exceptions.ConvergenceWarning)
            model.fit(X, y)

        # Save weights and biases
        weights.append(model.coef_)
        biases.append(model.intercept_)
    weights = np.stack(weights)
    biases = np.stack(biases)

    return weights, biases


def extract_features(feature_extractor, dset, avgpool=True, batch_size=256):
    """ Pass images in a dataset through a feature extractor.
    
    Arguments:
        feature_extractor (nn.Module): Receives an image and maps it to a feature vector.
        dsets (torch.Dataset): Dataset with (image, label) pairs.
        avgpool (bool): Whether to average any extra dimensions outputted by the feature 
            extractor, usually the spatial x, y dimensions.
        batch_size (int): Batch size to use when extracting features.
            
    Returns
        features (np.array): Returns the extracted features (num_examples x num_features).
    """
    import torch
    from torch.utils import data

    # Create dloader
    dloader = data.DataLoader(dset, batch_size=batch_size, num_workers=4)

    # Extract features
    features = []
    with torch.no_grad():
        for im, _ in dloader:
            feats = feature_extractor(im)
            if feats.ndim > 2 and avgpool:
                feats = feats.mean(dim=tuple(range(2, feats.ndim)))
            features.append(feats)
    features = torch.concat(features).cpu().numpy()

    return features


def compute_metrics_linear(weights, biases, X, y):
    """ Compute metrics for a linear model.
    
    If the weight vector has a single class assumes binary classification (using sigmoid 
    to predict probabilities for the positive class); in multi-class setting, softmax is
    used.
    
    Arguments:
        weights (np.array): Coefficients of the linear model (num_classes x num_features).
        biases (np.array): Bias (num_classes) 
        X (np.array): Dataset (num_examples x num_features)
        y (np.array): Targets (num_examples)
    
    Returns
        metrics (np.array): Array with metrics (as returned by utils.compute_metrics)
    """
    from scipy import special

    linear_output = np.dot(X, weights.T) + biases # num_examples x num_classes
    if linear_output.shape[-1] == 1: # binary case
        pos_probs = special.expit(linear_output) # sigmoid
        probs = np.concatenate([1-pos_probs, pos_probs], axis=-1)
    else:
        probs = special.softmax(linear_output, axis=-1)

    # Compute metrics
    metrics = np.array(utils.compute_metrics(probs, y))

    return metrics


def train_linear_on_resnet(train_dset, val_dset, save_dir):
    """ Train linear models on top of ResNet features.
    
    Images are send throught the resnet and normalized. Trains models across 4 resnet 
    depths/blocks and a number of regularization strengths, evaluates them on training and
    validation set and saves the results in an h5 file.
    
    Arguments:
        train_dset, val_dset (torch.Dataset): Datasets with images, labels. Will be send
            to the ResNet to extract features.
        save_dir (string): Path to the folder where the trained models (and training/val 
            metrics) will be saved.
    """
    resnet_blocks = [1, 2, 3, 4]
    reg_values = np.logspace(-4, 4, 17)
    results = []
    for resnet_block in resnet_blocks:
        utils.tprint(f'Creating ResNet features for resnet_block {resnet_block}')

        # Get ResNet
        resnet = models.ResNetBase(num_blocks=resnet_block, pretrained=True)
        resnet.eval()
        #resnet.cuda()

        # Get resnet features
        train_features = extract_features(resnet, train_dset)
        val_features = extract_features(resnet, val_dset)

        # Normalize features
        train_mean, train_std = train_features.mean(0), train_features.std(0)
        train_features = (train_features - train_mean) / train_std
        val_features = (val_features - train_mean) / train_std

        # Get labels
        train_labels = train_dset.labels
        val_labels = val_dset.labels

        # Train models
        utils.tprint('Training linear models...')
        weights, biases = train_linear(train_features, train_labels, reg_values)

        # Evaluate
        utils.tprint('Evaluating...')
        train_metrics = np.stack([
            compute_metrics_linear(w, b, train_features, train_labels)
            for w, b in zip(weights, biases)])
        val_metrics = np.stack([
            compute_metrics_linear(w, b, val_features, val_labels)
            for w, b in zip(weights, biases)])

        # Save results
        results.append({
            'weights': weights, 'biases': biases, 'train_metrics': train_metrics,
            'val_metrics': val_metrics})
    train_metrics = np.stack([r['train_metrics'] for r in results])  # num_resnets x num_regs x num_metrics
    val_metrics = np.stack([r['val_metrics'] for r in results])  # num_resnets x num_regs x num_metrics

    # Save
    utils.tprint('Saving models...')
    with h5py.File(path.join(save_dir, 'linear_on_resnet.h5'), 'w') as f:
        f.create_dataset('resnet_blocks', data=resnet_blocks, dtype=np.int)
        f.create_dataset('reg_values', data=reg_values)
        f.create_dataset('train_metrics', data=train_metrics)
        f.create_dataset('val_metrics', data=val_metrics)
        for i, res in enumerate(results):
            f.create_dataset(f'{i}/weights', data=res['weights'])
            f.create_dataset(f'{i}/biases', data=res['biases'])


def train_linear_on_human(train_dset, val_dset, abl_model, save_dir):
    """ Train a linear model on top of predicted human attributes.
        
    AbL is used to obtain the probabilities per attributes; these are concatenated into a 
    single vector of probs. Trains models across a number of regularization strengths, 
    evaluates them on training and validation set and saves the results in an h5 file.
    
    Arguments:
        train_dset, val_dset (torch.Dataset): Datasets with images, labels. Will be send
            to the ResNet to extract features.
        abl_model (nn.Module): Attribute predictor model. Receives an image, predicts a 
            list with the pre-softmax logits per attribute.
        save_dir (string): Path to the folder where the trained models (and training/val 
            metrics) will be saved.
    """
    from torch import nn

    utils.tprint('Predicting human attributes...')
    abl_model = nn.Sequential(abl_model, models.SoftmaxPlusConcat())
    abl_model.eval()
    #abl_model.cuda()

    # Get features
    train_features = extract_features(abl_model, train_dset)
    val_features = extract_features(abl_model, val_dset)

    # Get labels
    train_labels = train_dset.labels
    val_labels = val_dset.labels

    # Train models
    utils.tprint('Training linear models...')
    reg_values = np.logspace(-4, 4, 17)
    weights, biases = train_linear(train_features, train_labels, reg_values)

    # Evaluate
    utils.tprint('Evaluating...')
    train_metrics = np.stack([compute_metrics_linear(w, b, train_features, train_labels)
        for w, b in zip(weights, biases)])
    val_metrics = np.stack([compute_metrics_linear(w, b, val_features, val_labels)
        for w, b in zip(weights, biases)])

    # Save
    utils.tprint('Saving models...')
    with h5py.File(path.join(save_dir, 'linear_on_human.h5'), 'w') as f:
        f.create_dataset('reg_values', data=reg_values)
        f.create_dataset('train_metrics', data=train_metrics)
        f.create_dataset('val_metrics', data=val_metrics)
        f.create_dataset('weights', data=weights)
        f.create_dataset('biases', data=biases)


def train_linear_on_joint(train_dset, val_dset, abl_model, save_dir):
    """ Train a linear model on top of resnet_features AND predicted human attributes.
        
    Concatenates resnet features (after normalization) with the predicted human attribute
    probabilities. Trains models across different resnet depths/blocks and a number of 
    regularization strengths, evaluates them on training and validation set and saves the
    results in an h5 file.
    
    Arguments:
        train_dset, val_dset (torch.Dataset): Datasets with images, labels. Will be send
            to the ResNet to extract features.
        abl_model (nn.Module): Attribute predictor model. Receives an image, predicts a 
            list with the pre-softmax logits per attribute.
        save_dir (string): Path to the folder where the trained models (and training/val 
            metrics) will be saved.
    """
    from torch import nn

    utils.tprint('Predicting human attributes...')
    abl_model = nn.Sequential(abl_model, models.SoftmaxPlusConcat())
    abl_model.eval()
    #abl_model.cuda()

    # Get features
    train_human_features = extract_features(abl_model, train_dset)
    val_human_features = extract_features(abl_model, val_dset)

    # Get resnet features and train models on the concatenation of features
    resnet_blocks = [1, 2, 3, 4]
    reg_values = np.logspace(-4, 4, 17)
    results = []
    for resnet_block in resnet_blocks:
        utils.tprint(f'Creating ResNet features for resnet_block {resnet_block}')

        # Get ResNet
        resnet = models.ResNetBase(num_blocks=resnet_block, pretrained=True)
        resnet.eval()
        #resnet.cuda()

        # Get resnet features
        train_resnet_features = extract_features(resnet, train_dset)
        val_resnet_features = extract_features(resnet, val_dset)

        # # Normalize features
        train_mean = train_resnet_features.mean(0)
        train_std = train_resnet_features.std(0)
        train_resnet_features = (train_resnet_features - train_mean) / train_std
        val_resnet_features = (val_resnet_features - train_mean) / train_std

        # Create joint features
        train_features = np.concatenate([train_human_features, train_resnet_features], -1)
        val_features = np.concatenate([val_human_features, val_resnet_features], -1)

        # Get labels
        train_labels = train_dset.labels
        val_labels = val_dset.labels

        # Train models
        utils.tprint('Training linear models...')
        weights, biases = train_linear(train_features, train_labels, reg_values)

        # Evaluate
        utils.tprint('Evaluating...')
        train_metrics = np.stack([
            compute_metrics_linear(w, b, train_features, train_labels)
            for w, b in zip(weights, biases)])
        val_metrics = np.stack([
            compute_metrics_linear(w, b, val_features, val_labels)
            for w, b in zip(weights, biases)])

        # Save results
        results.append({
            'weights': weights, 'biases': biases, 'train_metrics': train_metrics,
            'val_metrics': val_metrics})
    train_metrics = np.stack([r['train_metrics']
                              for r in results])  # num_resnets x num_regs x num_metrics
    val_metrics = np.stack([r['val_metrics']
                            for r in results])  # num_resnets x num_regs x num_metrics

    # Save
    utils.tprint('Saving models...')
    with h5py.File(path.join(save_dir, 'linear_on_joint.h5'), 'w') as f:
        f.create_dataset('resnet_blocks', data=resnet_blocks, dtype=np.int)
        f.create_dataset('reg_values', data=reg_values)
        f.create_dataset('train_metrics', data=train_metrics)
        f.create_dataset('val_metrics', data=val_metrics)
        for i, res in enumerate(results):
            f.create_dataset(f'{i}/weights', data=res['weights'])
            f.create_dataset(f'{i}/biases', data=res['biases'])



# Set directory to save results
DDSM_dir = '/src/dermosxai/data/DDSM/classifiers'
HAM10000_dir = '/src/dermosxai/data/HAM10000/classifiers'


def train_DDSM_linear_on_resnet():
    """ Train linear models on the resnet features for the DDSM dataset."""
    # Get dsets
    utils.tprint('Getting datasets')
    train_dset = datasets.DDSM('train')
    val_dset = datasets.DDSM('val')

    # Add transforms
    _, val_transform = transforms.get_DDSM_transforms(train_dset.img_mean,
                                                      train_dset.img_std, make_rgb=True)
    train_dset.transform = val_transform
    val_dset.transform = val_transform

    train_linear_on_resnet(train_dset, val_dset, DDSM_dir)


def train_HAM10000_linear_on_resnet():
    """ Train linear models on the resnet features for the HAM10000 dataset."""
    # Get dsets
    utils.tprint('Getting datasets')
    train_dset = datasets.HAM10000('train')
    val_dset = datasets.HAM10000('val')

    # Add transforms
    _, val_transform = transforms.get_HAM10000_transforms(train_dset.img_mean,
                                                          train_dset.img_std)
    train_dset.transform = val_transform
    val_dset.transform = val_transform

    train_linear_on_resnet(train_dset, val_dset, HAM10000_dir)


def train_DDSM_linear_on_human():
    """ Train linear models on top of predicted human attributes. """
    # Get dsets
    utils.tprint('Getting datasets')
    train_dset = datasets.DDSM('train')
    val_dset = datasets.DDSM('val')

    # Add transforms
    _, val_transform = transforms.get_DDSM_transforms(train_dset.img_mean,
                                                      train_dset.img_std, make_rgb=True)
    train_dset.transform = val_transform
    val_dset.transform = val_transform

    # Get attribute predictor
    abl_model = train_abl.get_DDSM_AbL()

    train_linear_on_human(train_dset, val_dset, abl_model, DDSM_dir)


def train_HAM10000_linear_on_human():
    """ Train linear models on top of predicted human attributes. """
    # Get dsets
    utils.tprint('Getting datasets')
    train_dset = datasets.HAM10000('train')
    val_dset = datasets.HAM10000('val')

    # Add transforms
    _, val_transform = transforms.get_HAM10000_transforms(train_dset.img_mean,
                                                          train_dset.img_std)
    train_dset.transform = val_transform
    val_dset.transform = val_transform

    # Get attribute predictor
    abl_model = train_abl.get_HAM10000_AbL()

    train_linear_on_human(train_dset, val_dset, abl_model, HAM10000_dir)


def train_DDSM_linear_on_joint():
    """ Train linear models on top of resnet features + predicted human attributes. """
    # Get dsets
    utils.tprint('Getting datasets')
    train_dset = datasets.DDSM('train')
    val_dset = datasets.DDSM('val')

    # Add transforms
    _, val_transform = transforms.get_DDSM_transforms(train_dset.img_mean,
                                                      train_dset.img_std, make_rgb=True)
    train_dset.transform = val_transform
    val_dset.transform = val_transform

    # Get attribute predictor
    abl_model = train_abl.get_DDSM_AbL()

    train_linear_on_joint(train_dset, val_dset, abl_model, DDSM_dir)


def train_HAM10000_linear_on_joint():
    """ Train linear models on top of resnet fatures + predicted human attributes. """
    # Get dsets
    utils.tprint('Getting datasets')
    train_dset = datasets.HAM10000('train')
    val_dset = datasets.HAM10000('val')

    # Add transforms
    _, val_transform = transforms.get_HAM10000_transforms(train_dset.img_mean,
                                                          train_dset.img_std)
    train_dset.transform = val_transform
    val_dset.transform = val_transform

    # Get attribute predictor
    abl_model = train_abl.get_HAM10000_AbL()

    train_linear_on_joint(train_dset, val_dset, abl_model, HAM10000_dir)


################ Evaluation code
def _find_best_resnet_model(h5_path):
    with h5py.File(h5_path, 'r') as f:
        val_mcc = f['val_metrics'][..., 2]
        i, j = np.unravel_index(val_mcc.argmax(), val_mcc.shape)
        best_weight = f[f'{i}/weights'][j]
        best_bias = f[f'{i}/biases'][j]
        best_resnet_block = f['resnet_blocks'][i]
        best_reg_value = f['reg_values'][j]

    return best_weight, best_bias, (best_resnet_block, best_reg_value)


def eval_DDSM_linear_on_resnet():
    """ Test metrics for the best linear_on_resnet model."""
    # Load test set
    train_dset = datasets.DDSM('train')  # need this for normalization
    test_dset = datasets.DDSM('test')

    # Add transforms
    _, val_transform = transforms.get_DDSM_transforms(train_dset.img_mean,
                                                      train_dset.img_std, make_rgb=True)
    train_dset.transform = val_transform
    test_dset.transform = val_transform

    # Find the best model
    h5_path = path.join(DDSM_dir, 'linear_on_resnet.h5')
    weight, bias, (resnet_block, _) = _find_best_resnet_model(h5_path)

    # Get resnet
    resnet = models.ResNetBase(num_blocks=resnet_block, pretrained=True)
    resnet.eval()

    # Get resnet features
    train_features = extract_features(resnet, train_dset)
    test_features = extract_features(resnet, test_dset)

    # Normalize
    train_mean, train_std = train_features.mean(0), train_features.std(0)
    test_features = (test_features - train_mean) / train_std

    # Evaluate
    metrics = compute_metrics_linear(weight, bias, test_features, test_dset.labels)

    print('Test metrics (resnet, DDSM):', metrics)


def eval_HAM10000_linear_on_resnet():
    """ Test metrics for the best linear_on_resnet model."""
    # Load test set
    train_dset = datasets.HAM10000('train')  # need this for normalization
    test_dset = datasets.HAM10000('test')

    # Add transforms
    _, val_transform = transforms.get_HAM10000_transforms(train_dset.img_mean,
                                                          train_dset.img_std)
    train_dset.transform = val_transform
    test_dset.transform = val_transform

    # Find the best model
    h5_path = path.join(HAM10000_dir, 'linear_on_resnet.h5')
    weight, bias, (resnet_block, _) = _find_best_resnet_model(h5_path)

    # Get resnet
    resnet = models.ResNetBase(num_blocks=resnet_block, pretrained=True)
    resnet.eval()

    # Get resnet features
    train_features = extract_features(resnet, train_dset)
    test_features = extract_features(resnet, test_dset)

    # Normalize
    train_mean, train_std = train_features.mean(0), train_features.std(0)
    test_features = (test_features - train_mean) / train_std

    # Evaluate
    metrics = compute_metrics_linear(weight, bias, test_features, test_dset.labels)

    print('Test metrics (resnet, HAM10000):', metrics)


def eval_DDSM_linear_on_human():
    """ Test metrics for the best linear_on_human model."""
    from torch import nn

    # Load test set
    train_dset = datasets.DDSM('train')  # need this for normalization
    test_dset = datasets.DDSM('test')

    # Add transforms
    _, val_transform = transforms.get_DDSM_transforms(train_dset.img_mean,
                                                      train_dset.img_std, make_rgb=True)
    train_dset.transform = val_transform
    test_dset.transform = val_transform

    # Get attribute learner
    abl_model = train_abl.get_DDSM_AbL()
    abl_model = nn.Sequential(abl_model, models.SoftmaxPlusConcat())
    abl_model.eval()

    # Get features
    test_features = extract_features(abl_model, test_dset)

    # Find the best model
    h5_path = path.join(DDSM_dir, 'linear_on_human.h5')
    with h5py.File(h5_path, 'r') as f:
        val_mcc = f['val_metrics'][..., 2]
        weight = f['weights'][val_mcc.argmax()]
        bias = f['biases'][val_mcc.argmax()]

    # Evaluate
    metrics = compute_metrics_linear(weight, bias, test_features, test_dset.labels)

    print('Test metrics (human, DDSM):', metrics)


def eval_HAM10000_linear_on_human():
    """ Test metrics for the best linear_on_human model."""
    from torch import nn

    # Load test set
    train_dset = datasets.HAM10000('train')  # need this for normalization
    test_dset = datasets.HAM10000('test')

    # Add transforms
    _, val_transform = transforms.get_HAM10000_transforms(train_dset.img_mean,
                                                          train_dset.img_std)
    train_dset.transform = val_transform
    test_dset.transform = val_transform

    # Get attribute learner
    abl_model = train_abl.get_HAM10000_AbL()
    abl_model = nn.Sequential(abl_model, models.SoftmaxPlusConcat())
    abl_model.eval()

    # Get features
    test_features = extract_features(abl_model, test_dset)

    # Find the best model
    h5_path = path.join(HAM10000_dir, 'linear_on_human.h5')
    with h5py.File(h5_path, 'r') as f:
        val_mcc = f['val_metrics'][..., 2]
        weight = f['weights'][val_mcc.argmax()]
        bias = f['biases'][val_mcc.argmax()]

    # Evaluate
    metrics = compute_metrics_linear(weight, bias, test_features, test_dset.labels)

    print('Test metrics (human, HAM10000):', metrics)


def eval_DDSM_linear_on_joint():
    """ Test metrics for the best linear_on_joint model."""
    from torch import nn

    # Load test set
    train_dset = datasets.DDSM('train')  # need this for normalization
    test_dset = datasets.DDSM('test')

    # Add transforms
    _, val_transform = transforms.get_DDSM_transforms(train_dset.img_mean,
                                                      train_dset.img_std, make_rgb=True)
    train_dset.transform = val_transform
    test_dset.transform = val_transform

    # Get attribute learner
    abl_model = train_abl.get_DDSM_AbL()
    abl_model = nn.Sequential(abl_model, models.SoftmaxPlusConcat())
    abl_model.eval()

    # Get features
    test_human_features = extract_features(abl_model, test_dset)

    # Find the best model
    h5_path = path.join(DDSM_dir, 'linear_on_joint.h5')
    weight, bias, (resnet_block, _) = _find_best_resnet_model(h5_path)

    # Get resnet
    resnet = models.ResNetBase(num_blocks=resnet_block, pretrained=True)
    resnet.eval()

    # Get resnet features
    train_resnet_features = extract_features(resnet, train_dset)
    test_resnet_features = extract_features(resnet, test_dset)

    # Normalize
    train_mean = train_resnet_features.mean(0)
    train_std = train_resnet_features.std(0)
    test_resnet_features = (test_resnet_features - train_mean) / train_std

    # Concatenate features
    test_features = np.concatenate([test_human_features, test_resnet_features], -1)

    # Evaluate
    metrics = compute_metrics_linear(weight, bias, test_features, test_dset.labels)

    print('Test metrics (joint, DDSM):', metrics)


def eval_HAM10000_linear_on_joint():
    """ Test metrics for the best linear_on_joint model."""
    from torch import nn

    # Load test set
    train_dset = datasets.HAM10000('train')  # need this for normalization
    test_dset = datasets.HAM10000('test')

    # Add transforms
    _, val_transform = transforms.get_HAM10000_transforms(train_dset.img_mean,
                                                          train_dset.img_std)
    train_dset.transform = val_transform
    test_dset.transform = val_transform

    # Get attribute learner
    abl_model = train_abl.get_HAM10000_AbL()
    abl_model = nn.Sequential(abl_model, models.SoftmaxPlusConcat())
    abl_model.eval()

    # Get features
    test_human_features = extract_features(abl_model, test_dset)

    # Find the best model
    h5_path = path.join(HAM10000_dir, 'linear_on_joint.h5')
    weight, bias, (resnet_block, _) = _find_best_resnet_model(h5_path)

    # Get resnet
    resnet = models.ResNetBase(num_blocks=resnet_block, pretrained=True)
    resnet.eval()

    # Get resnet features
    train_resnet_features = extract_features(resnet, train_dset)
    test_resnet_features = extract_features(resnet, test_dset)

    # Normalize
    train_mean = train_resnet_features.mean(0)
    train_std = train_resnet_features.std(0)
    test_resnet_features = (test_resnet_features - train_mean) / train_std

    # Concatenate features
    test_features = np.concatenate([test_human_features, test_resnet_features], -1)

    # Evaluate
    metrics = compute_metrics_linear(weight, bias, test_features, test_dset.labels)

    print('Test metrics (joint, HAM10000):', metrics)


if __name__ == '__main__':
    """ For reference. Takes around 40 mins"""
    # Train all models
    train_DDSM_linear_on_resnet()
    train_DDSM_linear_on_human()
    train_DDSM_linear_on_joint()
    train_HAM10000_linear_on_resnet()
    train_HAM10000_linear_on_human()
    train_HAM10000_linear_on_joint()

    # Test all models (this just prints stuff)
    eval_DDSM_linear_on_resnet()
    eval_DDSM_linear_on_human()
    eval_DDSM_linear_on_joint()
    print()
    eval_HAM10000_linear_on_resnet()
    eval_HAM10000_linear_on_human()
    eval_HAM10000_linear_on_joint()

"""
Test metrics (resnet, DDSM): [0.72727273 0.44052187 0.44359055 0.67206478 0.78299092 0.72841511]
Test metrics (human, DDSM): [0.74074074 0.45628967 0.45639209 0.6695279  0.80807453 0.73024843]
Test metrics (joint, DDSM): [0.68686869 0.34539637 0.34561048 0.60425532 0.73884376 0.66322116]

Test metrics (resnet, HAM10000): [0.87411598 0.67191378 0.67273709 0.7323828  0.94910305 0.77070733]
Test metrics (human, HAM10000): [0.78925035 0.3352286  0.3610032  0.4917989  0.8627192  0.55125827]
Test metrics (joint, HAM10000): [0.85148515 0.62600131 0.62618153 0.70036548 0.9366731  0.74395133]
"""

def compute_DDSM_joint_MI():
    """Computes mutual information between resnet features and predicted DDSM attributes."""
    import torch
    from torch import nn

    # Load test set
    train_dset = datasets.DDSM('train')
    val_dset = datasets.DDSM('val')
    test_dset = datasets.DDSM('test')

    # Add transforms
    _, val_transform = transforms.get_DDSM_transforms(train_dset.img_mean,
                                                      train_dset.img_std, make_rgb=True)
    train_dset.transform = val_transform
    val_dset.transform = val_transform
    test_dset.transform = val_transform

    # Get attribute learner
    abl_model = train_abl.get_DDSM_AbL()
    abl_model = nn.Sequential(abl_model, models.SoftmaxPlusConcat())
    abl_model.eval()

    # Get features
    train_human_features = extract_features(abl_model, train_dset)
    val_human_features = extract_features(abl_model, val_dset)
    test_human_features = extract_features(abl_model, test_dset)

    # Find the best resnet_layer
    h5_path = path.join(DDSM_dir, 'linear_on_joint.h5')
    _, _, (resnet_block, _) = _find_best_resnet_model(h5_path)

    # Get resnet
    resnet = models.ResNetBase(num_blocks=resnet_block, pretrained=True)
    resnet.eval()

    # Get resnet features
    train_resnet_features = extract_features(resnet, train_dset)
    val_resnet_features = extract_features(resnet, val_dset)
    test_resnet_features = extract_features(resnet, test_dset)

    # Normalize
    train_mean = train_resnet_features.mean(0)
    train_std = train_resnet_features.std(0)
    train_resnet_features = (train_resnet_features - train_mean) / train_std
    val_resnet_features = (val_resnet_features - train_mean) / train_std
    test_resnet_features = (test_resnet_features - train_mean) / train_std

    # Make them tensors
    train_human_features = torch.tensor(train_human_features)
    val_human_features = torch.tensor(val_human_features)
    test_human_features = torch.tensor(test_human_features)
    train_resnet_features = torch.tensor(train_resnet_features)
    val_resnet_features = torch.tensor(val_resnet_features)
    test_resnet_features = torch.tensor(test_resnet_features)

    # Train MI estimator
    from dermosxai import mi
    mi_estimator = mi.train_mi(train_human_features, train_resnet_features,
                               val_human_features, val_resnet_features)[0]

    # Compute MI
    mi_estimator.eval()
    with torch.no_grad():
        dv, _, jsd, infonce = mi_estimator(test_human_features, test_resnet_features)

    print('Test MI (DDSM): ', dv.item(), jsd.item(), infonce.item())


def compute_HAM10000_joint_MI():
    """Computes mutual information between resnet features and predicted DDSM attributes."""
    import torch
    from torch import nn

    # Load test set
    train_dset = datasets.HAM10000('train')
    val_dset = datasets.HAM10000('val')
    test_dset = datasets.HAM10000('test')

    # Add transforms
    _, val_transform = transforms.get_HAM10000_transforms(train_dset.img_mean,
                                                          train_dset.img_std)
    train_dset.transform = val_transform
    val_dset.transform = val_transform
    test_dset.transform = val_transform

    # Get attribute learner
    abl_model = train_abl.get_HAM10000_AbL()
    abl_model = nn.Sequential(abl_model, models.SoftmaxPlusConcat())
    abl_model.eval()

    # Get features
    train_human_features = extract_features(abl_model, train_dset)
    val_human_features = extract_features(abl_model, val_dset)
    test_human_features = extract_features(abl_model, test_dset)

    # Find the best resnet_layer
    h5_path = path.join(HAM10000_dir, 'linear_on_joint.h5')
    _, _, (resnet_block, _) = _find_best_resnet_model(h5_path)

    # Get resnet
    resnet = models.ResNetBase(num_blocks=resnet_block, pretrained=True)
    resnet.eval()

    # Get resnet features
    train_resnet_features = extract_features(resnet, train_dset)
    val_resnet_features = extract_features(resnet, val_dset)
    test_resnet_features = extract_features(resnet, test_dset)

    # Normalize
    train_mean = train_resnet_features.mean(0)
    train_std = train_resnet_features.std(0)
    train_resnet_features = (train_resnet_features - train_mean) / train_std
    val_resnet_features = (val_resnet_features - train_mean) / train_std
    test_resnet_features = (test_resnet_features - train_mean) / train_std

    # Make them tensors
    train_human_features = torch.tensor(train_human_features)
    val_human_features = torch.tensor(val_human_features)
    test_human_features = torch.tensor(test_human_features)
    train_resnet_features = torch.tensor(train_resnet_features)
    val_resnet_features = torch.tensor(val_resnet_features)
    test_resnet_features = torch.tensor(test_resnet_features)

    # Train MI estimator
    from dermosxai import mi
    mi_estimator = mi.train_mi(train_human_features, train_resnet_features,
                               val_human_features, val_resnet_features)[0]

    # Compute MI
    mi_estimator.eval()
    with torch.no_grad():
        dv, _, jsd, infonce = mi_estimator(test_human_features, test_resnet_features)

    print('Test MI (HAM10000): ', dv.item(), jsd.item(), infonce.item())

"""
Test MI (DDSM):  0.31409886479377747 -1.252334713935852 -5.379660129547119

Test MI (HAM10000):  3.1195478439331055 -0.5205932855606079 -3.4473540782928467
"""
