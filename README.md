# Semantic Segmentation Engine 🚗🌆

A modular and customizable semantic segmentation engine built with **PyTorch**, **Segmentation Models PyTorch (SMP)**, **Albumentations**, and **TensorBoard**. This project is tailored for training, validating, and testing segmentation models, with support for datasets like **Cityscapes**.

## 🔍 Features

- 📂 Custom `Dataset` class for loading image-mask pairs
- 🧠 Supports popular segmentation architectures (e.g., DeepLabV3+, Unet, FPN via SMP)
- 🔁 Training/Validation/Test split with metric tracking
- 📊 TensorBoard support for visual logs
- 🔄 Augmentation support via `albumentations`
- 🛠️ Easily extendable and cleanly modularized code

### 📁 Project Structure

```
Semantic-Segmentation-engine/
├── dataset/          # Custom Dataset class (e.g., for Cityscapes)
├── engine/           # Training, validation, and test loops
├── config/           # Configuration files (model, dataset, training)
├── utils/            # Utility functions (metrics, visualizations)
├── main.py           # Entry point for training and evaluation
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Atrin-Dev/Semantic-Segmentation-engine.git
cd Semantic-Segmentation-engine
```

---

## ⚙️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```
### 🧠 Available Models

You can choose between the following **architectures**:

- `DeepLabV3+`
- `UNet`

**Backbones** can be any supported encoder from [Segmentation Models PyTorch (SMP)](https://github.com/qubvel/segmentation_models.pytorch), such as:

- `resnet34`
- `efficientnet-b4`
- `mobilenet_v2`
- *(and many more)*

.

## 🚀 Usage
### Train a model
```bash
python main.py --model deeplabv3plus --encoder efficientnet-b4 --weights imagenet --mode train
```
### Test a trained model
```bash
python main.py --model unet --encoder resnet34 --weights imagenet --mode test
```
### Train and test together
```bash
python main.py --model deeplabv3plus --encoder resnet50 --weights imagenet --mode all
```
## 🛠️ Configuration
Edit global training settings such as batch size, epochs, and data paths in config.py.

Example:

```python
DATA_PATH = "/path/to/cityscapes"
BATCH_SIZE = 4
MAX_EPOCHS = 60
DEVICE = "cuda"  # or "cpu"
NUM_CLASSES = 20
```

