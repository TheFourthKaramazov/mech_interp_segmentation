# Universality Hypothesis in Deep SOTA Segmentation Models: CNN vs Transformers

This repository investigates the **universality hypothesis** in deep segmentation models by comparing the internal representations of two state-of-the-art models: **UNet++** (a CNN-based architecture) and **Mask2Former** (a Transformer-based architecture). Our goal is to determine whether these models, when trained on the same binary segmentation task using the ISIC 2018 dataset, converge to similar internal representations despite their fundamentally different designs.

We combine both quantitative similarity metrics—such as Linear CKA, SVCCA, Kernel CKA, and RSA—with qualitative visualization techniques (including PCA projections) to perform a comprehensive analysis. Our experiments focus on both encoder and decoder layers:

- **Encoder layers** (frozen during fine-tuning) exhibit moderate alignment (e.g., Linear CKA ≈ 0.35 for UNet++'s `encoder.layer3` versus Mask2Former's corresponding encoder layer), suggesting some commonality resulting from pretraining.
- **Decoder layers** (fine-tuned on the segmentation task) reveal very low linear similarity and significant divergence when measured with subspace metrics, highlighting how architectural biases dominate representation formation even when both models are tasked with the same objective.

Despite modest nonlinear similarities indicated by metrics such as Kernel CKA, our PCA-based visualizations further underscore that the spatial activation patterns differ considerably between architectures. While the encoders show some regionally localized features in the CNN and globally mixed patterns in the Transformer, the decoders are even more distinct—UNet++ decoders exhibit cohesive, task-specific activation regions, whereas Mask2Former decoders are more fragmented.

The repository also explores potential projection techniques (e.g., canonical correlation analysis, Procrustes analysis, and manifold alignment) that may help project the representations into a common latent space for more meaningful comparisons.

## Repository Structure

```
├── dataset.py                    # Dataset loading and preprocessing
├── experiments.py                # Feature extraction, similarity metrics, and visualization functions (all mechanistic interpretability experiment functions are in this notebook)
├── inference.py                  # Inference utilities for testing
├── loss_metrics.py               # Segmentation metrics and losses (Scaled Dice, IoU, etc.)
├── mask2former.ipynb             # Mask2Former-specific training and analysis notebook
├── mech_interp_experiment.ipynb  # Main experiment and analysis notebook for mechanistic interpretability (bulk of project here)
├── train.py                      # Model training functions
├── unetpp.ipynb                  # UNet++-specific training and analysis notebook
├── utils.py                      # Helper utilities
├── visualization.py              # Data visualization for pre-processing, traning, and testing
├── .gitignore
├── requirements.txt              # Python dependencies (to be completed)
└── README.md                     # This file
```
## Models & Data

Due to their large size, the dataset and pre-trained models are not included in the repository directly. Instead, please use the following Google Drive links to download the required files:

- **Preprocessed ISIC 2018 Dataset (Binary Segmentation Task):** [Download Dataset (Google Drive)](https://drive.google.com/file/d/1V09WzvSrCrUuZs8u-pmbH-1WVcBhwwlp/view?usp=share_link)
- **Pretrained UNet++ and Mask2Former Checkpoints:** [Download Models (Google Drive)](https://drive.google.com/drive/folders/1hXUsWBKZdr9SpN_NGXoQNnxYNQ4i4twI?usp=sharing)

After downloading, place the files into the appropriate directories (`data/data` and `models/`) at the root of the repository.
```
**Data Directory Structure**

├── data
   ├── data
      ├── ISIC2018_Test_GroundTruth
      ├── ISIC2018_Test_Input
      ├── ISIC2018_Training_GroundTruth
      ├── ISIC2018_Training_Input
      ├── ISIC2018_Validation_GroundTruth
      ├── ISIC2018_Validation_Input


**Models Directory Structure**

├── models
   ├── unet
      ├── unetpp_epoch10_batch16 
   ├── transformer
      ├── mask2former_epoch10_batch16
```


## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/unetpp-mask2former-universality.git
   cd unetpp-mask2former-universality
   ```
2. **Set up a virtual environment and install dependencies:**

   ```bash
   python -m venv venv
   source venv/bin/activate        # For Linux/Mac
   # or
   venv\Scripts\activate           # For Windows

   pip install -r requirements.txt


   ```

## Usage

### Training and Fine-tuning

**Detailed Instructions Provided throughout `mask2former.ipynb`and  `unetpp.ipynb`**

> Note: **The training notebooks are executed by default but will have their longer outputs (visualization and inference qualitative results contianing 1000 image, GT mask, predicted mask comparisons hidden if not opened in COLAB)**

- These notebooks were built for google colab for ease of use. Instructions are provided throughout the main three notebooks: `mask2former.ipynb`, `unetpp.ipynb`, and `mech_interp_experiment.ipynb`,
- However, if you have a CUDA enabled machine, you can also ignore the google colab code and run the training scripts locally
- The mechanistic interpretability experimetns notebook `mech_interp_experiment.ipynb`, can be run locally on CPU if desired. The delay is not significant. Simply ignore the google mounting code.

### Experimentation and Analysis

- Open the main experiment notebook:
  `mech_interp_experiment.ipynb`

  This notebook includes:

  - Feature extraction from selected encoder and decoder layers.
  - Computation of similarity metrics (Linear CKA, SVCCA, Kernel CKA, RSA).
  - PCA-based visualization of activations.
  - Detailed analysis of representation similarities and differences.


### Results and Analysis

This notebook presents an in-depth investigation into the internal representations of two state-of-the-art segmentation models—Unet++ (a CNN-based architecture) and Mask2Former (a Transformer-based architecture)—to evaluate the universality hypothesis. The central objective was to determine whether models trained on the same segmentation task converge to similar internal representations despite their fundamentally different design principles. Both quantitative metrics and qualitative visualizations were employed to provide a comprehensive view.



### Overview of Results


##### Encoder Comparisons

| UNet++ Layer         | Mask2Former Layer                                                           | Linear CKA | SVCCA    |
|----------------------|-----------------------------------------------------------------------------|------------|----------|
| `encoder.layer3`     | `model.pixel_level_module.encoder.embeddings.norm`                         | 0.3569     | -0.0475  |
| `encoder.layer4`     | `model.pixel_level_module.encoder.encoder.layers.2.blocks.0.output.dense`  | 0.2382     | -0.0353  |

These encoder results indicate moderate linear similarity—likely due to the shared pre-training but also reveal inconsistent subspace alignment as evidenced by slightly negative SVCCA values.

##### Decoder Comparisons

| UNet++ Layer            | Mask2Former Layer                                              | Linear CKA | SVCCA    | Kernel CKA | RSA (Spearman) |
|-------------------------|----------------------------------------------------------------|------------|----------|------------|----------------|
| `decoder.blocks.x_1_1.conv1`  | `model.transformer_module.decoder.mask_predictor.mask_embedder.0.0` | 0.0682     | 0.0196   | 0.0818     | 0.0094         |
| `decoder.blocks.x_1_1.conv1`  | `model.transformer_module.decoder.mask_predictor.mask_embedder.1.0` | 0.0246     | -0.0420  | 0.0768     | -0.0270        |
| `decoder.blocks.x_1_1.conv1`  | `model.transformer_module.decoder.mask_predictor.mask_embedder.2.0` | 0.0516     | 0.0209   | 0.0602     | 0.0121         |
| `segmentation_head.0`  | `model.transformer_module.decoder.mask_predictor.mask_embedder.0.0`     | 0.0367     | nan      | 0.0349     | nan            |
| `segmentation_head.0`  | `model.transformer_module.decoder.mask_predictor.mask_embedder.1.0`     | 0.0010     | nan      | 0.0254     | nan            |
| `segmentation_head.0`  | `model.transformer_module.decoder.mask_predictor.mask_embedder.2.0`     | 0.0204     | nan      | 0.0114     | nan            |

The decoder comparisons demonstrate that the fine-tuned representations exhibit very low alignment in both linear and nonlinear measures. Even when a modest Kernel CKA (up to 0.0818) is observed for the intermediate decoder layers, the overall similarity remains minimal. Comparisons involving UNet++’s final output (`segmentation_head.0`) are especially poor.



#### Quantitative Analysis

Several similarity metrics were computed across selected layers of the two models:

- **Linear Metrics:**  
  - **Linear CKA:** Encoder comparisons showed a Linear CKA of approximately 0.3569 for Unet++’s `encoder.layer3` versus Mask2Former’s `model.pixel_level_module.encoder.embeddings.norm`, indicating moderate linear alignment in mid-level features. For higher-level encoder features (`encoder.layer4` vs. `model.pixel_level_module.encoder.encoder.layers.2.blocks.0.output.dense`), Linear CKA was around 0.2382. In contrast, decoder layers exhibited very low Linear CKA values—for example, UNet++’s `decoder.blocks.x_1_1.conv1` versus Mask2Former’s `mask_embedder.0.0` showed a value of approximately 0.0295—suggesting minimal linear similarity in task-specific representations.

- **Nonlinear Metrics:**  
  - **Kernel CKA:** Although nonlinear metric values were modest (e.g., around 0.0818 for the UNet++ `decoder.blocks.x_1_1.conv1` vs. Mask2Former’s `mask_embedder.0.0` comparison), these values indicate some shared structure within a nonlinear manifold.  
  - **SVCCA and RSA:** SVCCA results were near zero or even slightly negative in many cases (and undefined for comparisons involving extremely low-dimensional outputs such as `segmentation_head.0`), while RSA produced similarly weak correlations. Overall, both measures suggest that even advanced subspace and relational comparisons do not reveal strong alignment between the internal representations of these architectures.



### Final Performance Results and Implications

**UNet++ on the Test Set:**  
- **Mean IoU:** 0.8265  
- **Mean Dice:** 0.8983  
- **Mean Pixel Accuracy:** 0.9394  

**Mask2Former on the Test Set:**  
- **Mean IoU:** 0.8034  
- **Mean Dice:** 0.8835  
- **Mean Pixel Accuracy:** 0.9297  

Training performance indicated that both models reached similar overall metrics during fine-tuning, with strong performance on the segmentation task. However, further evaluation—particularly of the decoder layers which were fine-tuned (as opposed to the frozen encoder layers) showed a different picture. The quantitative analysis of the decoders showed very low linear and only modest nonlinear similarities, indicating minimal convergence in internal representations between the architectures.

These updated results confirm that even though both architectures perform comparably on the task (as reflected in similar quantitative performance), the internal features, especially in the task-specific decoders, remain largely divergent. This divergence implies that the expected universality—that similar tasks lead to similar internal representations—is not supported across architectures with large SOTA segmentation models. Advanced projection and alignment techniques may be required to further investigate and potentially bridge the latent representational gap between CNN-based and Transformer-based models.



#### Qualitative Visualization and PCA Exploration

PCA was applied to high-dimensional activation maps to project them into a 3-channel “RGB” space for visualization:

- **Encoder Visualizations:**  
  The PCA-based images for encoder layers revealed that Unet++ produces localized, block-like activations, indicative of strong spatial locality in convolutional operations. In contrast, the Transformer encoder activations appear more globally mixed, reflecting the influence of the self-attention mechanism. Although quantitative metrics suggest moderate similarity, the visual differences underscore distinct spatial encoding strategies.
  
- **Decoder Visualizations:**  
  The PCA visualizations of decoder layers further illustrate divergence. Unet++’s decoder activations, such as those from `decoder.blocks.x_1_1.conv1`, exhibit coherent, regionally defined patterns. On the other hand, the corresponding transformer decoder activations show more numerous and fragmented patches with lighter colors, implying that task-specific fine-tuning does not fully align the representations between the models.

The exploration with PCA demonstrated that while dimensionality reduction helps visualize the dominant variance in the activations, it does not capture the full complexity of the nonlinear transformations unique to each architecture. This indicates a need for more sophisticated projection techniques to map the representations into a common latent space.



#### Future Directions: Projection and Alignment Techniques

Based on the modest nonlinear alignment observed, further work should explore advanced projection techniques to better align the latent spaces of CNNs and Transformers. Approaches such as:
- **Canonical Correlation Analysis (CCA)**
- **Procrustes Analysis**
- **Manifold Alignment Techniques**

could provide a more direct comparison between the features learned by each model and reveal deeper, underlying similarities that raw metrics and PCA-based visualizations fail to capture.



#### Final Thoughts

The comprehensive analysis clearly demonstrates the absence of universality across architectures for large segmentation models. Despite both Unet++ and Mask2Former achieving comparable performance on the segmentation task, the internal representations do not converge to a similar structure. In fact, while the encoder layers frozen during fine-tuning exhibit only modest alignment, the decoder layers, which were fine-tuned on the same exact task, show very weak similarity. This result was unexpected: it was anticipated that the decoders, being task-specific and trained under identical conditions, would converge more strongly. Instead, the low linear similarity and modest nonlinear alignment reveal that inherent inductive biases in CNNs versus Transformers lead to fundamentally different ways of encoding and processing information.

This notebook serves as a detailed case study in mechanistic interpretability and provides clear evidence that there is no cross-architecture universality for these large segmentation models. It also lays the groundwork for future research aimed at bridging this representational gap through advanced alignment and projection techniques.


## Citation

If you find this work useful, please consider citing it:

@misc{Universality Hypothesis in Deep SOTA Segmentation Models: CNN vs Transformer,
title={Universality Hypothesis in Deep SOTA Segmentation Models: CNN vs Transformers},
author={Brandon Leblanc},
year={2025},
note={https://github.com/TheFourthKaramazov/mech_interp_segmentation}}
