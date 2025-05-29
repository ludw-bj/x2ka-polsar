# X2Ka-PolSAR: Conditional GAN for X-to-Ka Band PolSAR Image Translation

**Ka-band Polarimetric Synthetic Aperture Radar (PolSAR)** offers high-resolution imaging and compact hardware potential due to its wide bandwidth and millimeter wavelengths. However, the required high sampling rate increases data transfer demands and implementation costs, limiting its deployment. As a result, Ka-band PolSAR images remain scarce compared to more established bands like L-/C-/X-band.

This scarcity of real Ka-band PolSAR data presents a critical challenge for research and development. **Traditional SAR simulators** — both raw signal-based and image-based — struggle to address this issue due to either high complexity or poor fidelity in capturing Ka-band-specific scattering effects.

🚀 This project offers a new data-driven path to augmenting Ka-band PolSAR datasets using more readily available X-band data.

---

## 📌 Project Overview

This repository provides a **PyTorch implementation** of [X2Ka-PolSAR](https://ieeexplore.ieee.org/document/10282106): a **Conditional GAN (cGAN)** for *translating X-band PolSAR images into Ka-band equivalents*. It builds upon the well-established **Pix2Pix** frameworks to model the image translation task, and introduces a **patch-based statistical similarity loss** to better preserve polarimetric and statistical characteristics.

In addition to the original model, we introduce:
- **Perceptual loss-based pretraining** for enhanced training stability.
- A **CycleGAN variant** that enables unpaired image translation for scenarios lacking pixel-level correspondences.


---

## 📷 Example Results

![Example Result](result_img/example_output.png)

> *(Above: input X-band PolSAR image, predicted Ka-band image, and ground truth)*

---

## 📊 Quantitative Results

| **Metrics** | **L1** | **MSE** |
|-------------|--------|---------|
| **Baseline (Pix2Pix)** | 0.121 | 0.030 |
| **Proposed (X2Ka)** | **0.115** | **0.026** |

---

## 📁 Dataset

The network was trained on an X-/Ka-band PolSAR dataset acquired in **Hainan, China**, using the **Aerial Remote Sensing System** developed by the **Chinese Academy of Sciences**.

⚠️ **Due to data usage restrictions, the raw dataset and trained model parameters are not publicly released.**

---

## 🛠️ Dependencies

This project is built on top of:

- **[CycleGAN](https://junyanz.github.io/CycleGAN/)**: [Paper](https://arxiv.org/pdf/1703.10593.pdf) | [Code](https://github.com/junyanz/CycleGAN)
- **[Pix2Pix](https://phillipi.github.io/pix2pix/)**: [Paper](https://arxiv.org/pdf/1611.07004.pdf) | [Code](https://github.com/phillipi/pix2pix)

Refer to the official [PyTorch implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) for details on environment setup and baseline training.

---

## 🧠 Key Features

- ✅ Conditional GAN-based architecture for cross-frequency PolSAR translation
- ✅ Patch-based statistical similarity loss for improved structural consistency
- ✅ Extensible to other frequency bands or modalities

---

## 🚀 Getting Started

1. **Clone this repo**
    ```bash
    git clone https://github.com/ludw-bj/x2ka-polsar.git
    cd x2ka-polsar
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare your own dataset**
    - **Pix2Pix**: Organize your dataset as paired cross-frequency PolSAR image patches in the format {A, B}.
    - **CycleGAN**: Place images from the two frequency bands into separate folders:
    `[datapath]/A` for domain A (e.g., X-band) and `[datapath]/B` for domain B (e.g., Ka-band).
    > Example `.jpg`-formatted datasets are provided under `./datasets/x2ka_aligned` (for paired data) and `./datasets/x2ka_separate` (for unpaired data).
    > Due to data usage restrictions, the full raw dataset and pretrained model weights are not publicly available.

4. **Train the model**
    Use one of the training scripts in the `./scripts` directory. Example:
    ```bash
    python train.py --dataroot ./datasets/x2ka --name x2ka_pix2pix --model pix2pix --netG unet_256
    ```

5. **Test the model**
    Similarly, use the test script in the `./scripts` directory. Example:
    ```bash
    python test.py --dataroot ./datasets/x2ka --name x2ka_pix2pix --model pix2pix --netG unet_256
    ```

🔧 Customize other training and testing parameters as needed. See options/train_options.py and options/test_options.py for full configuration options.

---

## 📄 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@INPROCEEDINGS{x2ka-polsar,
  author={Lu, Danwei and Sun, Tianyu and Wang, Hongmiao and Yin, Junjun and Yang, Jian},
  booktitle={IGARSS 2023 - 2023 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={Adaptive Conditional GAN based Ka-Band PolSAR Image Simulation by Using X-Band PolSAR Image Transfer}, 
  year={2023},
  volume={},
  number={},
  pages={8082-8085},
  keywords={Adaptation models;Image resolution;Apertures;Information retrieval;Generative adversarial networks;Robustness;Polarimetric synthetic aperture radar;Polarimetric Synthetic Aperture Radar;conditional Generative Adversarial Network;data insufficiency;Ka-band;neural style transfer},
  doi={10.1109/IGARSS52108.2023.10282106}}

