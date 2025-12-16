# Sem-NeuS: Semantically-Guided High-Fidelity Neural Surface Reconstruction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)


This repository contains the official implementation of **Sem-NeuS**, a framework for high-fidelity 3D reconstruction that integrates **Semantic-Guided Geometry Distillation** with a hierarchical SDF architecture. By distilling features from Vision Foundation Models (DINOv3) into 3D geometry, our method resolves ambiguities in textureless and thin structures (e.g., insect limbs) without using semantic features as direct input.

<p align="center">
  <img src="assets/fig_1.png" alt="Sem-NeuS Teaser" width="100%">
</p>

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Sem-NeuS.git
cd Sem-NeuS
```
### 2. Environment Setup
We recommend using Anaconda to manage dependencies.
```bash
conda create -n semneus python=3.9
conda activate semneus
```
### 3. Install Dependencies
Install the required Python packages:
```
pip install -r requirements.txt
```

## 🐜 Dataset Download

We release the high-resolution **Entomological Dataset** presented in the paper, featuring challenging thin structures and glossy surfaces (e.g., *Oechalia cf. schellenbergi*). The data includes calibrated cameras (COLMAP) and segmentation masks (SAM).

| Dataset | Description | Download |
| :--- | :--- | :--- |
| **Insect-01** | *Oechalia cf. schellenbergi* (211 views) | [**Download (.zip)**](INSERT_YOUR_LINK_HERE) |
| **OmniObject3D** | Selected scenes (Statue, Lego, Bread) | [Official Website](https://omniobject3d.github.io/) |

### How to use
Your directory tree should look like this:

```
Sem-NeuS/
└── data/
    └── insect_01/          <-- Use this name for --case
        ├── image/
        ├── mask/
        └── cameras_sphere.npz
```

## 🦕 DINOv3 Model Setup
Our framework relies on a pre-trained DINOv3 model (ViT-L/16) to provide semantic guidance. You must download the weights before training.
Download the weights:
Download the dinov3_vitl16.pth checkpoint.
```
wget https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16.pth
```
Place the file:
Move the .pth file to the root directory of this repository. Your folder structure should look like this:
```
Sem-NeuS/
├── dinov3_vitl16.pth        <-- Place here
├── exp_runner.py
├── models/
│   ├── semantic/
│   └── ...
└── ...
```
Note: You do not need to manually extract features. The training script automatically detects if features are missing for a dataset and runs the extractor (models/semantic/preprocess.py) before training starts.
📂 Data Preparation
Organize your data following the standard NeuS format. For custom datasets (e.g., insects), use COLMAP to generate camera poses.
```
data/
└── <CASE_NAME>/
    ├── image/               # Input RGB images
    ├── mask/                # (Optional) Foreground masks
    ├── cameras_sphere.npz   # Camera parameters (NeuS format)
    └── dinov3_features/     # (Auto-generated )
```
🚀 Usage
### 1. Training
To train a scene, run exp_runner_high.py. The script will automatically extract DINO features if they don't exist.
```
python exp_runner.py \
    --mode train \
    --conf ./confs/womask.conf \
    --case <CASE_NAME>
```
--case: The name of your scene folder inside data/ (e.g., china_statue, insect_01).
--conf: Configuration file path.
### 2. Mesh Extraction (Validation)
To extract the mesh from a trained model using Marching Cubes:
```
python exp_runner.py \
    --mode validate_mesh \
    --conf ./confs/womask.conf \
    --case <CASE_NAME> \
    --is_continue \
    --mesh_resolution 1024
```
The output mesh will be saved in exp/<CASE_NAME>/meshes/.
### 3. Novel View Synthesis
To render novel views (interpolation) between two cameras:
python exp_runner.py \
    --mode interpolate_0_10 \
```
    --conf ./confs/womask_high_dtu.conf \
    --case <CASE_NAME> \
    --is_continue
