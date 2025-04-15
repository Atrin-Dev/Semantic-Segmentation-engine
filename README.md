# Semantic Segmentation Engine 🚗🌆

A modular and customizable semantic segmentation engine built with **PyTorch**, **Segmentation Models PyTorch (SMP)**, **Albumentations**, and **TensorBoard**. This project is tailored for training, validating, and testing segmentation models, with support for datasets like **Cityscapes**.

## 🔍 Features

- 📂 Custom `Dataset` class for loading image-mask pairs
- 🧠 Supports popular segmentation architectures (e.g., DeepLabV3+, Unet, FPN via SMP)
- 🔁 Training/Validation/Test split with metric tracking
- 📊 TensorBoard support for visual logs
- 🔄 Augmentation support via `albumentations`
- 🛠️ Easily extendable and cleanly modularized code

## 📁 Project Structure

<pre><code>``` Semantic-Segmentation-engine/ ├── dataset/ # Custom Dataset class (e.g., for Cityscapes) ├── engine/ # Training, validation, and test loops ├── config/ # Configuration files (model, dataset, training) ├── utils/ # Utility functions (metrics, visualizations) ├── main.py # Entry point for training and evaluation ├── requirements.txt # Python dependencies └── README.md # This file ``` </code></pre>

