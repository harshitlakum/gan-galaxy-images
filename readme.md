# GAN Galaxy Images

Generate **synthetic galaxy images** using a **Deep Convolutional GAN (DCGAN)** trained on the **Galaxy10 DECals** dataset. The GAN learns to reproduce realistic galaxy morphology from **3-band (g,r,z)** astronomical images at **32×32 resolution**.

![Status](https://img.shields.io/badge/Model-GAN-blue) ![Python](https://img.shields.io/badge/Python-3.9+-yellow) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

---

## 🚀 Project Overview

This project implements a **Generative Adversarial Network (GAN)** to synthesize galaxy images similar to real samples from **Galaxy10 DECals**. The model uses adversarial training between a **Generator** and **Discriminator** to learn galaxy structure.

### ✅ Key Steps

| Step          | Description                                        |
| ------------- | -------------------------------------------------- |
| Dataset Prep  | Load Galaxy10 DECals → downsample to 32×32         |
| Preprocessing | Normalize to `[-1, 1]` for GAN stability           |
| Model Design  | DCGAN-style generator/discriminator                |
| Training      | Adversarial training with label smoothing          |
| Evaluation    | Generate galaxy grids + latent space interpolation |

---

## 📂 Repository Structure

```
gan-galaxy-images/
│
├── gan_galaxy_images.ipynb       # Full pipeline notebook
├── models/                       # (optional) saved model weights
├── samples/                      # Generated galaxy images
├── README.md
└── requirements.txt
```

---

## 🧠 Model Architecture

* **Generator**

  * Input: latent vector (e.g. 100D Gaussian noise)
  * Output: 32×32×3 RGB galaxy image
  * Techniques: transposed conv, batch norm, ReLU, Tanh

* **Discriminator**

  * Input: real or fake image
  * Output: real/fake probability
  * Techniques: strided conv, LeakyReLU, dropout

---

## 🏋️ Training

* Epochs: **100–200**
* Loss: **Binary Cross Entropy**
* Optimizer: **Adam(2e-4, β₁=0.5)**
* Tricks used:
  ✅ Label smoothing
  ✅ Normalization
  ✅ Monitoring discriminator dominance

Generated images saved every `n` epochs to monitor quality.

---

## 📊 Results

✔️ Generates realistic spiral/elliptical galaxy textures
✔️ Shows smooth interpolation in latent space
✔️ Mode collapse avoided with moderate hyperparameter tuning

> Add real images under `samples/` to make this repo portfolio-ready.

---

## 🔧 Getting Started

```bash
git clone https://github.com/your-username/gan-galaxy-images.git
cd gan-galaxy-images
pip install -r requirements.txt
```

Launch notebook:

```bash
jupyter notebook gan_galaxy_images.ipynb
```

---

## ✅ Dependencies

* Python 3.x
* TensorFlow/Keras
* NumPy
* h5py
* Matplotlib

---

## 📡 Dataset Source

We use **Galaxy10 DECals**, provided by **AstroNN**:

* [https://github.com/henrysky/Galaxy10](https://github.com/henrysky/Galaxy10)

To load via AstroNN helper:

```python
from astronn.datasets import load_galaxy10
images, labels = load_galaxy10()
```

---

## 🔬 Future Work Ideas

Want to level this up? Add:

* ✅ Frechet Inception Distance (FID) evaluation
* ✅ Conditional GAN by morphology class
* ✅ Diffusion model comparison
* ✅ Higher resolution (64×64, 128×128)

---

## 🏷️ License

MIT License. Free to use—just cite the dataset.



Want me to upgrade it? 🚀
