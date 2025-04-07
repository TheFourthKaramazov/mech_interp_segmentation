###########
# Imports #
###########


import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import numpy as np
from sklearn.decomposition import TruncatedSVD
import torch.nn.functional as F
import scipy.spatial.distance as ssd
import scipy.stats as stats

class FeatureExtractor:
    """
    Class for extracting activations from specified layers using forward hooks.
    """
    def __init__(self, model, layers):
        """
        Initialize with model and a list of layer names.
        
        Parameters:
            model (nn.Module): The model from which to extract features.
            layers (list of str): List of attribute paths (e.g., "encoder.layer1") specifying the layers.
        """
        self.model = model
        self.layers = layers
        self.features = {}
        self.hooks = []
        self._register_hooks()
    
    
    def _get_module_by_name(self, module, access_string):
        """
        helper unction to access a module by its attribute path.

        Parameters: 
            module (nn.Module): The module to access.
            access_string (str): The attribute path to the desired module.
        Returns:
            nn.Module: The desired module
        """
        # recursively access module by attribute path
        names = access_string.split('.')
        for name in names:
            module = getattr(module, name)
        return module
    
    def _hook_fn(self, layer_name):
        """
        helper function to create a hook function that stores the activations.

        Parameters:
            layer_name (str): The name of the layer to store activations for.
        Returns:
            function: The hook function
        """
        # closure to store the activations
        def hook(module, input, output):
            if layer_name not in self.features:
                self.features[layer_name] = []
            # detach and store output
            self.features[layer_name].append(output.detach().cpu())
        return hook
    
    
    def _register_hooks(self):
        """
        helper function to register hooks for each layer.
        """
        # register a forward hook for each layer
        for layer_name in self.layers:
            module = self._get_module_by_name(self.model, layer_name)
            hook = module.register_forward_hook(self._hook_fn(layer_name))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """
        function to remove all hooks.
        """

        for hook in self.hooks:
            hook.remove()

def extract_features(model, dataloader, layers, device='cuda'):
    """
    Extract activations from specified layers over the entire dataset.
    
    Parameters:
        model (nn.Module): The model for which to extract features.
        dataloader (DataLoader): DataLoader providing input images.
        layers (list of str): List of layer names (as in FeatureExtractor) to extract.
        device (str): Device to run inference on.
    
    Returns:
        features (dict): Dictionary mapping layer names to a concatenated tensor of activations.
    """
    model.eval()
    extractor = FeatureExtractor(model, layers) # create a FeatureExtractor

    # iterate over the dataset and extract features
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Extracting Features"):
            inputs = inputs.to(device) # move inputs to device
            
            _ = model(inputs) # forward pass to extract features
    features = {}

    # concatenate the activations for each layer
    for layer, acts in extractor.features.items():
        features[layer] = torch.cat(acts, dim=0)

    # remove hooks before returning
    extractor.remove_hooks()

    # return the features dictionary mapping layer names to activations
    return features 

def subsample_tensor(tensor, n_samples=1000):
    """
    Randomly subsample the first dimension of the tensor if needed.
    
    Parameters:
        tensor (torch.Tensor): Input tensor with shape (n_samples, ...).
        n_samples (int): Number of samples to keep.
    
    Returns:
        torch.Tensor: Subsampled tensor.
    """
    if tensor.size(0) > n_samples:
        indices = torch.randperm(tensor.size(0))[:n_samples]
        return tensor[indices]
    else:
        return tensor

def compute_linear_cka(X, Y):
    """
    Compute the linear CKA similarity between two sets of activations.
    
    Parameters:
        X (torch.Tensor): Activations with shape (n_samples, n_features).
        Y (torch.Tensor): Activations with shape (n_samples, n_features).
    
    Returns:
        float: The linear CKA similarity score.
    """
    # center the activations before computing the similarity
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    
    # compute the linear CKA similarity and normalize by the norms
    dot_product_similarity = (X.T @ Y).norm() ** 2
    normalization_x = (X.T @ X).norm() ** 2
    normalization_y = (Y.T @ Y).norm() ** 2
    
    # add a small value to the denominator to prevent division by zero
    return dot_product_similarity / (torch.sqrt(normalization_x * normalization_y) + 1e-6)

def compute_svcca_truncated(X, Y, k=20):
    """
    Compute a basic SVCCA similarity using truncated SVD between two sets of activations.

    we truncate because we only want to keep the top k singular vectors to avoid large memory
    spikes that can take more than 50GB on RAM due to the large size of flattened vectors in mask2former
    after the norm module in the transformer. 

    
    Parameters:
        X (torch.Tensor): Activations with shape (n_samples, n_features).
        Y (torch.Tensor): Activations with shape (n_samples, n_features).
        k (int): Number of singular vectors to keep.
    
    Returns:
        float: The SVCCA similarity score, or np.nan if not computable.
    """
    # convert tensors to numpy arrays
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    
    # find the number of components to use
    n_features = X_np.shape[1]
    k_eff = min(k, n_features)
    
    # if less than 2 features, return NaN as SVCCA is not computable
    if k_eff < 2:
        return np.nan
    
    # use truncated SVD to reduce the dimensionality
    svd = TruncatedSVD(n_components=k_eff)
    X_top = svd.fit_transform(X_np)
    Y_top = svd.fit_transform(Y_np)
    
    # compute the correlations between the top components
    correlations = []
    for i in range(k_eff):
        corr = np.corrcoef(X_top[:, i], Y_top[:, i])[0, 1]
        correlations.append(corr)
    
    return np.mean(correlations) # return the mean correlation


def pool_activations(tensor, output_size=1):
    """
    Apply adaptive average pooling to reduce spatial or sequential dimensions.
    
    If tensor is 4D (batch, channels, height, width), it uses adaptive_avg_pool2d.
    If tensor is 3D (batch, channels, length), it uses adaptive_avg_pool1d.
    Otherwise, returns the tensor unchanged.
    
    we truncate because we only want to keep the top k singular vectors to avoid large memory
    spikes that can take more than 50GB on RAM due to the large size of flattened vectors in mask2former
    after the norm module in the transformer. 

    Parameters:
        tensor (torch.Tensor): The input activation tensor.
        output_size (int or tuple): The desired output size for the spatial/sequence dimension.
    
    Returns:
        torch.Tensor: The pooled tensor.
    """
    if tensor.dim() == 4:
        # for 4D tensors, expect output_size as tuple or int (converted to tuple)
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        return F.adaptive_avg_pool2d(tensor, output_size)
    
    elif tensor.dim() == 3:
        # for 3D tensors, use adaptive_avg_pool1d; output_size should be an int.
        if isinstance(output_size, tuple):
            output_size = output_size[0]
        return F.adaptive_avg_pool1d(tensor, output_size)
    else:
        return tensor

DEBUG = True # helps with memory spikes (avoid disconnecting GPU and losing local variables)


def compare_layer_pair(features_unet, features_mask2former, layer_unet, layer_mask2former,
                       n_samples=1000, k_svcca=20):
    """
    Compare the activations of one layer from UNet++ and one layer from Mask2Former.

    Parameters:
        features_unet (dict): Dictionary of UNet++ activations.
        features_mask2former (dict): Dictionary of model activations.
        layer_unet (str): Layer name in UNet++.
        layer_mask2former (str): Layer name in model.
        n_samples (int): Number of samples to use for comparison.
        k_svcca (int): Number of singular vectors to keep for SVCCA.

    Returns:
        tuple: Pair of similarity scores
    """
    # extract activations for the given layers
    acts_unet = features_unet[layer_unet]
    acts_mask2former = features_mask2former[layer_mask2former]
    
    # apply pooling to reduce spatial or sequential dimensions to avoid memory spikes
    acts_unet = pool_activations(acts_unet, output_size=1)
    acts_mask2former = pool_activations(acts_mask2former, output_size=1)
    
    # subsample activations (this will not force them to have the same number, so we'll adjust later)
    acts_unet = subsample_tensor(acts_unet, n_samples=n_samples)
    acts_mask2former = subsample_tensor(acts_mask2former, n_samples=n_samples)
    
    # debug prints for tensor sizes
    if DEBUG:
        print(f"[DEBUG] {layer_unet} shape after subsampling: {acts_unet.shape}")
        print(f"[DEBUG] {layer_mask2former} shape after subsampling: {acts_mask2former.shape}")
    
    # ensure both tensors have the same number of samples
    n1 = acts_unet.size(0)
    n2 = acts_mask2former.size(0)
    n_common = min(n1, n2)
    if n1 != n_common:
        acts_unet = subsample_tensor(acts_unet, n_samples=n_common)
    if n2 != n_common:
        acts_mask2former = subsample_tensor(acts_mask2former, n_samples=n_common)
    
    # flatten the activations so they have shape (n_common, n_features)
    # this is where we can get serious memory spikes
    X = acts_unet.view(acts_unet.size(0), -1)
    Y = acts_mask2former.view(acts_mask2former.size(0), -1)
    
    if DEBUG: # debug prints for flattened tensor sizes
        print(f"[DEBUG] {layer_unet} flattened shape: {X.shape}")
        print(f"[DEBUG] {layer_mask2former} flattened shape: {Y.shape}")
    
    # compute similarity metrics
    cka_score = compute_linear_cka(X, Y)
    svcca_score = compute_svcca_truncated(X, Y, k=k_svcca)
    

    return cka_score, svcca_score

def print_layer_comparison(features_unet, features_mask2former, layer_unet, layer_mask2former,
                           n_samples=1000, k_svcca=20):
    """
    Compare layers and print the similarity metrics.

    Parameters:
        features_unet (dict): Dictionary of UNet++ activations.
        features_mask2former (dict): Dictionary of Mask2Former activations.
        layer_unet (str): Layer name in UNet++.
        layer_mask2former (str): Layer name in Mask2Former.
        n_samples (int): Number of samples to use for comparison.
        k_svcca (int): Number of singular vectors to keep for

    Returns:
        None

    """

    # compare the layers
    cka_score, svcca_score = compare_layer_pair(features_unet, features_mask2former, layer_unet,
                                                 layer_mask2former, n_samples=n_samples, k_svcca=k_svcca)
    
    # print the results
    print(f"Comparing UNet++ '{layer_unet}' vs Mask2Former '{layer_mask2former}':")
    print(f"  Linear CKA: {cka_score:.4f}")
    print(f"  SVCCA: {svcca_score:.4f}\n")

def investigate_candidate_layers(model, candidate_layers, dataloader, device):
    """
    Run one batch through the model to extract and print activation shapes for candidate layers.
    
    Parameters:
        model (nn.Module): The model (either UNet++ or Mask2Former).
        candidate_layers (list of str): List of candidate layer attribute paths.
        dataloader (DataLoader): A dataloader to provide a single batch.
        device (torch.device): The computation device.

    Returns:
        None
    """
    # use the FeatureExtractor to hook onto candidate layers
    extractor = FeatureExtractor(model, candidate_layers)
    model.eval()

    # run one batch through the model to extract activations
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Extracting features for candidate layers"):
            inputs = inputs.to(device)
            try:
                _ = model(pixel_values=inputs)
            except TypeError:
                _ = model(inputs)
            break  # only one batch needed

    # print the shapes of the extracted activations for each candidate layer
    print("Candidate Layers and their Activation Shapes:")
    for layer in tqdm(candidate_layers, desc="Investigating candidate layers", unit="layer", bar_format="{l_bar}{bar}| {remaining}"):
        acts = torch.cat(extractor.features[layer], dim=0)
        print(f"Layer '{layer}': shape {acts.shape}")
    
    # remove hooks after investigation
    extractor.remove_hooks()

def compute_kernel_cka(X, Y, sigma=None):
    """
    Compute a normalized nonlinear (kernel) CKA similarity using an RBF kernel.

    Parameters:
        X (np.array): Activations with shape (n_samples, n_features).
        Y (np.array): Activations with shape (n_samples, n_features).
        sigma (float): RBF kernel width. If None, uses the median pairwise distance.

    Returns:
        float: Kernel CKA similarity in [0, 1].
    """
    # estimate sigma if not provided
    if sigma is None:
        dists = ssd.pdist(X, 'euclidean')
        sigma = np.median(dists)
        if sigma == 0:
            sigma = 1.0

    # RBF kernel function
    def rbf_kernel(A):
        dists = ssd.squareform(ssd.pdist(A, 'sqeuclidean'))
        return np.exp(-dists / (2 * sigma**2))

    # center kernel matrix
    def center_kernel(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    # compute centered RBF kernels
    KX = center_kernel(rbf_kernel(X))
    KY = center_kernel(rbf_kernel(Y))

    # compute normalized HSIC (CKA)
    numerator = np.sum(KX * KY)
    denominator = np.sqrt(np.sum(KX * KX) * np.sum(KY * KY))

    return numerator / (denominator + 1e-6)  # avoid division by zero

def compute_rdm(X, metric='correlation'):
    """
    Compute the Representational Dissimilarity Matrix (RDM) for activations X.
    
    Parameters:
        X (np.array): Activations with shape (n_samples, n_features).
        metric (str): Distance metric (e.g., 'correlation', 'euclidean').
        
    Returns:
        np.array: A square RDM of shape (n_samples, n_samples).
    """
    return ssd.squareform(ssd.pdist(X, metric=metric))

def compare_rdm_similarity(X, Y, metric='correlation'):
    """
    Compare two RDMs by computing the Spearman correlation between their upper triangular parts.
    
    Parameters:
        X (np.array): Activations with shape (n_samples, n_features).
        Y (np.array): Activations with shape (n_samples, n_features).
        metric (str): Distance metric for RDM computation.
        
    Returns:
        float: Spearman correlation between the flattened upper triangular parts of the RDMs.
    """
    rdm_X = compute_rdm(X, metric=metric)
    rdm_Y = compute_rdm(Y, metric=metric)
    # get the upper triangular part of the RDMs
    idx = np.triu_indices_from(rdm_X, k=1)
    rdm_X_flat = rdm_X[idx]
    rdm_Y_flat = rdm_Y[idx]

    # compute the Spearman correlation between the flattened RDMs
    return stats.spearmanr(rdm_X_flat, rdm_Y_flat).correlation


