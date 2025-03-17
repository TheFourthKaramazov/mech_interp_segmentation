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
        # recursively access module by attribute path
        names = access_string.split('.')
        for name in names:
            module = getattr(module, name)
        return module
    
    def _hook_fn(self, layer_name):
        def hook(module, input, output):
            if layer_name not in self.features:
                self.features[layer_name] = []
            # detach and store output
            self.features[layer_name].append(output.detach().cpu())
        return hook
    
    def _register_hooks(self):
        for layer_name in self.layers:
            module = self._get_module_by_name(self.model, layer_name)
            hook = module.register_forward_hook(self._hook_fn(layer_name))
            self.hooks.append(hook)
    
    def remove_hooks(self):
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
    extractor = FeatureExtractor(model, layers)
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Extracting Features"):
            inputs = inputs.to(device)
            # This example assumes your model is called with a keyword "pixel_values"
            _ = model(inputs)
    features = {}
    for layer, acts in extractor.features.items():
        features[layer] = torch.cat(acts, dim=0)
    extractor.remove_hooks()
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
    # Center the activations
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    
    dot_product_similarity = (X.T @ Y).norm() ** 2
    normalization_x = (X.T @ X).norm() ** 2
    normalization_y = (Y.T @ Y).norm() ** 2
    
    return dot_product_similarity / (torch.sqrt(normalization_x * normalization_y) + 1e-6)

def compute_svcca_truncated(X, Y, k=20):
    """
    Compute a basic SVCCA similarity using truncated SVD between two sets of activations.
    
    Parameters:
        X (torch.Tensor): Activations with shape (n_samples, n_features).
        Y (torch.Tensor): Activations with shape (n_samples, n_features).
        k (int): Number of singular vectors to keep.
    
    Returns:
        float: The SVCCA similarity score, or np.nan if not computable.
    """
    # Convert tensors to numpy arrays
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    
    # Determine the effective number of features
    n_features = X_np.shape[1]
    k_eff = min(k, n_features)
    
    # If there is less than 2 features, SVCCA is not defined; return NaN
    if k_eff < 2:
        return np.nan
    
    # Use TruncatedSVD with the effective number of components
    svd = TruncatedSVD(n_components=k_eff)
    X_top = svd.fit_transform(X_np)
    Y_top = svd.fit_transform(Y_np)
    
    correlations = []
    for i in range(k_eff):
        corr = np.corrcoef(X_top[:, i], Y_top[:, i])[0, 1]
        correlations.append(corr)
    
    return np.mean(correlations)


def pool_activations(tensor, output_size=1):
    """
    Apply adaptive average pooling to reduce spatial or sequential dimensions.
    
    If tensor is 4D (batch, channels, height, width), it uses adaptive_avg_pool2d.
    If tensor is 3D (batch, channels, length), it uses adaptive_avg_pool1d.
    Otherwise, returns the tensor unchanged.
    
    Parameters:
        tensor (torch.Tensor): The input activation tensor.
        output_size (int or tuple): The desired output size for the spatial/sequence dimension.
    
    Returns:
        torch.Tensor: The pooled tensor.
    """
    if tensor.dim() == 4:
        # For 4D tensors, expect output_size as tuple or int (converted to tuple)
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        return F.adaptive_avg_pool2d(tensor, output_size)
    elif tensor.dim() == 3:
        # For 3D tensors, use adaptive_avg_pool1d; output_size should be an int.
        if isinstance(output_size, tuple):
            output_size = output_size[0]
        return F.adaptive_avg_pool1d(tensor, output_size)
    else:
        return tensor

DEBUG = True

def compare_layer_pair(features_unet, features_mask2former, layer_unet, layer_mask2former,
                       n_samples=1000, k_svcca=20):
    """
    Compare the activations of one layer from UNet++ and one layer from Mask2Former.
    """
    # Extract activations for the given layers
    acts_unet = features_unet[layer_unet]
    acts_mask2former = features_mask2former[layer_mask2former]
    
    # Apply pooling to reduce spatial or sequential dimensions
    acts_unet = pool_activations(acts_unet, output_size=1)
    acts_mask2former = pool_activations(acts_mask2former, output_size=1)
    
    # Subsample activations (this will not force them to have the same number, so we'll adjust later)
    acts_unet = subsample_tensor(acts_unet, n_samples=n_samples)
    acts_mask2former = subsample_tensor(acts_mask2former, n_samples=n_samples)
    
    # Debug prints
    if DEBUG:
        print(f"[DEBUG] {layer_unet} shape after subsampling: {acts_unet.shape}")
        print(f"[DEBUG] {layer_mask2former} shape after subsampling: {acts_mask2former.shape}")
    
    # Ensure both tensors have the same number of samples
    n1 = acts_unet.size(0)
    n2 = acts_mask2former.size(0)
    n_common = min(n1, n2)
    if n1 != n_common:
        acts_unet = subsample_tensor(acts_unet, n_samples=n_common)
    if n2 != n_common:
        acts_mask2former = subsample_tensor(acts_mask2former, n_samples=n_common)
    
    # Flatten the activations so they have shape (n_common, n_features)
    X = acts_unet.view(acts_unet.size(0), -1)
    Y = acts_mask2former.view(acts_mask2former.size(0), -1)
    
    if DEBUG:
        print(f"[DEBUG] {layer_unet} flattened shape: {X.shape}")
        print(f"[DEBUG] {layer_mask2former} flattened shape: {Y.shape}")
    
    # Compute similarity metrics
    cka_score = compute_linear_cka(X, Y)
    svcca_score = compute_svcca_truncated(X, Y, k=k_svcca)
    
    return cka_score, svcca_score

def print_layer_comparison(features_unet, features_mask2former, layer_unet, layer_mask2former,
                           n_samples=1000, k_svcca=20):
    """
    Compare layers and print the similarity metrics.
    """
    cka_score, svcca_score = compare_layer_pair(features_unet, features_mask2former, layer_unet,
                                                 layer_mask2former, n_samples=n_samples, k_svcca=k_svcca)
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
    """
    # Use the FeatureExtractor to hook onto candidate layers
    extractor = FeatureExtractor(model, candidate_layers)
    model.eval()
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Extracting features for candidate layers"):
            inputs = inputs.to(device)
            try:
                _ = model(pixel_values=inputs)
            except TypeError:
                _ = model(inputs)
            break  # Only one batch needed

    print("Candidate Layers and their Activation Shapes:")
    for layer in tqdm(candidate_layers, desc="Investigating candidate layers", unit="layer", bar_format="{l_bar}{bar}| {remaining}"):
        acts = torch.cat(extractor.features[layer], dim=0)
        print(f"Layer '{layer}': shape {acts.shape}")
    
    extractor.remove_hooks()

def compute_kernel_cka(X, Y, sigma=None):
    """
    Compute a nonlinear (kernel) CKA similarity using an RBF kernel.
    
    Parameters:
        X (np.array): Activations with shape (n_samples, n_features).
        Y (np.array): Activations with shape (n_samples, n_features).
        sigma (float): Kernel width parameter. If None, use the median heuristic.
        
    Returns:
        float: Kernel CKA similarity score.
    """
    # Estimate sigma if not provided using the median heuristic
    if sigma is None:
        sigma = np.median(ssd.pdist(X, metric='euclidean'))
        if sigma == 0:
            sigma = 1.0
    # Compute squared Euclidean distance matrices
    dist_X = ssd.squareform(ssd.pdist(X, metric='sqeuclidean'))
    dist_Y = ssd.squareform(ssd.pdist(Y, metric='sqeuclidean'))
    KX = np.exp(-dist_X / (2 * sigma**2))
    KY = np.exp(-dist_Y / (2 * sigma**2))
    
    # Center the kernel matrices
    def center_kernel(K):
        n = K.shape[0]
        one_n = np.ones((n, n)) / n
        return K - one_n @ K - K @ one_n + one_n @ K @ one_n

    KX_centered = center_kernel(KX)
    KY_centered = center_kernel(KY)
    
    numerator = np.linalg.norm(KX_centered @ KY_centered, 'fro')**2
    denominator = np.linalg.norm(KX_centered, 'fro')**2 * np.linalg.norm(KY_centered, 'fro')**2
    return numerator / (np.sqrt(denominator) + 1e-6)

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
    # Get upper triangular indices (excluding diagonal)
    idx = np.triu_indices_from(rdm_X, k=1)
    rdm_X_flat = rdm_X[idx]
    rdm_Y_flat = rdm_Y[idx]
    return stats.spearmanr(rdm_X_flat, rdm_Y_flat).correlation


