# Universality Hypothesis in Deep SOTA Segmentation Models: CNN vs Transformers

This repository investigates the **universality hypothesis** in deep segmentation models by comparing the internal representations of two state-of-the-art models: **UNet++** (a CNN-based architecture) and **Mask2Former** (a Transformer-based architecture). Our goal is to determine whether these models, when trained on the same binary segmentation task using the ISIC 2018 dataset, converge to similar internal representations despite their fundamentally different designs.

We combine both quantitative similarity metrics—such as Linear CKA, SVCCA, Kernel CKA, and RSA—with qualitative visualization techniques (including PCA projections) to perform a comprehensive analysis. Our experiments focus on both encoder and decoder layers:
- **Encoder layers** (frozen during fine-tuning) exhibit moderate alignment (e.g., Linear CKA ≈ 0.35 for UNet++'s `encoder.layer3` versus Mask2Former's corresponding encoder layer), suggesting some commonality resulting from pretraining.
- **Decoder layers** (fine-tuned on the segmentation task) reveal very low linear similarity and significant divergence when measured with subspace metrics, highlighting how architectural biases dominate representation formation even when both models are tasked with the same objective.

Despite modest nonlinear similarities indicated by metrics such as Kernel CKA, our PCA-based visualizations further underscore that the spatial activation patterns differ considerably between architectures. While the encoders show some regionally localized features in the CNN and globally mixed patterns in the Transformer, the decoders are even more distinct—UNet++ decoders exhibit cohesive, task-specific activation regions, whereas Mask2Former decoders are more fragmented.

The repository also explores potential projection techniques (e.g., canonical correlation analysis, Procrustes analysis, and manifold alignment) that may help project the representations into a common latent space for more meaningful comparisons.

## Repository Structure

├── dataset.py                    # Dataset loading and preprocessing

├── experiments.py                # Feature extraction, similarity metrics, and visualization functions

├── inference.py                  # Inference utilities for testing

├── loss_metrics.py               # Segmentation metrics and losses (Dice, IoU, etc.)

├── mask2former.ipynb             # Mask2Former-specific training and analysis notebook

├── mech_interp_experiment.ipynb  # Main experiment notebook for mechanistic interpretability

├── train.py                      # Model training functions

├── unetpp.ipynb                  # UNet++-specific analysis notebook

├── utils.py                      # Helper utilities 

├── visualization.py              # Data visualization for pre-processing, traning, and testing

├── .gitignore

├── requirements.txt              # Python dependencies (to be completed)

└── README.md                     # This file

## Models & Data

Due to their large size, the dataset and pre-trained models are not included in the repository directly. Instead, please use the following Google Drive links to download the required files:
- **Preprocessed ISIC 2018 Dataset (Binary Segmentation Task):** [Download Dataset (Google Drive)](https://drive.google.com/your-dataset-link-here)
- **Pretrained UNet++ and Mask2Former Checkpoints:** [Download Models (Google Drive)](https://drive.google.com/your-models-link-here)

After downloading, place the files into the appropriate directories (`data/` and `models/`) at the root of the repository.

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
   

## Usage

### Training and Fine-tuning

**Detailed Instructions Provided throughout `mask2former.ipynb`and  `unetpp.ipynb`**

> Note: **The training notebooks are executed by default but will have their longer outputs (visualization and inference qualitative results contianing 1000 image, GT mask, predicted mask comparisons hidden if not opened in COLAB)**

-  These notebooks were built for google colab for ease of use. Instructions are provided throughout the main three notebooks: `mask2former.ipynb`, `unetpp.ipynb`, and `mech_interp_experiment.ipynb`,
-  However, if you have a CUDA enabled machine, you can also ignore the google colab code and run the training scripts locally
- The mechanistic interpretability experimetns notebook `mech_interp_experiment.ipynb`, can be run locally on CPU if desired. The delay is not significant. Simply ignore the google mounting code.
  
### Experimentation and Analysis
- Open the main experiment notebook:
  `mech_interp_experiment.ipynb`

  This notebook includes:
  - Feature extraction from selected encoder and decoder layers.
  - Computation of similarity metrics (Linear CKA, SVCCA, Kernel CKA, RSA).
  - PCA-based visualization of activations.
  - Detailed analysis of representation similarities and differences.


## Citation

If you find this work useful, please consider citing it:

@misc{Universality Hypothesis in Deep SOTA Segmentation Models: CNN vs Transformer,
title={Universality Hypothesis in Deep SOTA Segmentation Models: CNN vs Transformers},
author={Brandon Leblanc},
year={2025},
note={https://github.com/TheFourthKaramazov/mech_interp_segmentation}}
