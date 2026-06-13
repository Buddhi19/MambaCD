# Precision Spatio-Temporal Feature Fusion for Robust Remote Sensing Change Detection
Official implementation.

This repository extends the ChangeMamba / VMamba-based change detection pipeline with **precision spatio-temporal fusion blocks**, an **enhanced lightweight decoder**, and an **IoU-aware optimization strategy** for robust binary remote sensing change detection.

---

## 🔥 Updates

* ✅ Paper is now live on IEEE Xplore: https://ieeexplore.ieee.org/document/11450773
* ✅ arXiv version: https://arxiv.org/abs/2507.11523
* ✅ Pretrained models released for **LEVIR-CD+**, **SYSU-CD**, and **WHU-CD**
* ✅ Code and inference notebook are available in this repository

---

## 🚀 Core Idea

Remote sensing change detection is not only a feature extraction problem. It is fundamentally a **bitemporal feature fusion problem**.

VMamba-style vision models process visual information through sequential state-space scanning. This makes them efficient for long-range spatial context modeling, but in change detection the model receives two temporally related images:

* **T1**: pre-change reference image
* **T2**: post-change current image

A key challenge is preserving the temporal identity of both feature streams. If T1 and T2 features are mixed too early or too uniformly, the network may dilute the actual change evidence. This can lead to missed small structures, noisy boundaries, and false positives caused by illumination, seasonal variation, shadows, or registration errors.

Our design follows a stronger fusion principle:

> **Use T1 as the reference memory, but keep T2 as the dominant post-change representation.**

The role of fusion is therefore not to simply concatenate or average the two timestamps. Instead, the fusion module should use the pre-change features to condition, contrast, and refine the post-change features so that the final representation becomes more sensitive to real structural change.

---

## 📊 Results

The method is evaluated on three widely used remote sensing change detection datasets:

* **LEVIR-CD+**
* **SYSU-CD**
* **WHU-CD**

Pretrained checkpoints are provided below.

| Dataset   | IoU (%) | Pretrained Model                                                                                  |
| --------- | ------: | ------------------------------------------------------------------------------------------------- |
| LEVIR-CD+ |   83.32 | [Download](https://drive.google.com/file/d/1uX0Yo8ov7EWlQr_EWxEM9UOwhdscY7M1/view?usp=drive_lin)  |
| SYSU-CD   |   75.04 | [Download](https://drive.google.com/file/d/1e6irCYcRmmtC2GPxEdAFd9j0LXd4HWnw/view?usp=drive_link) |
| WHU-CD    |   89.95 | [Download](https://drive.google.com/file/d/1JJPZNxF9-3KhQyrQ5z6xuuHFVRD8hRlx/view?usp=drive_link) |

Qualitative results:

![Qualitative Results](docs/image.png)

---

## 🧩 Architecture Overview

The model is built around a VMamba / ChangeMamba-style backbone and improved with precision fusion and decoder refinement.

![Modeling Mechanisms](docs/decoder_ICIIS.jpg)

**Main components:**

* VMamba-based feature extraction
* precision spatio-temporal fusion blocks
* explicit difference modeling
* CBAM-enhanced decoder refinement
* lightweight local-detail reconstruction

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/Buddhi19/MambaCD.git
cd MambaCD
```

Create and activate a Conda environment:

```bash
conda create -n mamba_cd python=3.10 -y
conda activate mamba_cd
```

Install PyTorch according to your CUDA version from the official PyTorch website:

https://pytorch.org/get-started/locally/

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

---

## 🎯 Quick Start

### Training

Run:

```bash
python train.py
```

Please update the dataset paths and training configuration according to your local setup before training.

### Inference

For inference and visualization steps, refer to:

```text
annotations/Ours.ipynb
```

---

## 📁 Repository Structure

```text
MambaCD/
├── changedetection/        # Change detection models and training components
├── classification/         # VMamba-related backbone components
├── kernels/                # Selective scan kernels
├── docs/                   # Figures and qualitative results
├── annotations/            # Inference / visualization notebooks
├── train.py                # Training entry point
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 📌 Paper Links

* IEEE Xplore: https://ieeexplore.ieee.org/document/11450773
* arXiv: https://arxiv.org/abs/2507.11523
* PDF: https://arxiv.org/pdf/2507.11523
* Code: https://github.com/Buddhi19/MambaCD
* Papers with Code: https://paperswithcode.com/paper/precision-spatio-temporal-feature-fusion-for

---

## 📚 Citation

If you find this work useful for your research, please cite:

```bibtex
@INPROCEEDINGS{11450773,
  author={Wijenayake, W.M.B.S.K. and Ratnayake, R.M.A.M.B. and Sumanasekara, D.M.U.P. and Wasalathilaka, N.S. and Piratheepan, M. and Godaliyadda, G.M.R.I. and Ekanayake, M.P.B. and Herath, H.M.V.R.},
  booktitle={2025 IEEE 19th International Conference on Industrial and Information Systems (ICIIS)}, 
  title={Precision Spatio-Temporal Feature Fusion for Robust Remote Sensing Change Detection}, 
  year={2026},
  volume={19},
  number={},
  pages={557-562},
  keywords={Accuracy;Computational modeling;Pipelines;Feature extraction;Transformers;Decoding;Remote sensing;Optimization;Monitoring;Context modeling;Remote Sensing;Binary Change Detection;State Space Models;Mamba},
  doi={10.1109/ICIIS69028.2026.11450773}
}
```

You may also cite our related experimental work on CBAM and Dice-loss-based semantic change detection:

```bibtex
@INPROCEEDINGS{11217111,
  author={Ratnayake, R.M.A.M.B. and Wijenayake, W.M.B.S.K. and Sumanasekara, D.M.U.P. and Godaliyadda, G.M.R.I. and Herath, H.M.V.R. and Ekanayake, M.P.B.},
  booktitle={2025 Moratuwa Engineering Research Conference (MERCon)}, 
  title={Enhanced SCanNet with CBAM and Dice Loss for Semantic Change Detection}, 
  year={2025},
  volume={},
  number={},
  pages={84-89},
  keywords={Training;Accuracy;Attention mechanisms;Sensitivity;Semantics;Refining;Feature extraction;Transformers;Power capacitors;Remote sensing},
  doi={10.1109/MERCon67903.2025.11217111}
}
```

---

## 🙏 Acknowledgments

This work builds on the excellent ChangeMamba codebase:

https://github.com/ChenHongruixuan/ChangeMamba

We sincerely thank the authors for releasing their implementation and enabling further research on Mamba-based remote sensing change detection.

---

## ⭐ Support

If this repository helps your research, please consider giving it a star..
