Vision Transformer (ViT) for Paddy Leaf Disease Classification

A from-scratch PyTorch implementation trained on real agricultural images

📌 Overview

This project presents a fully custom implementation of a Vision Transformer (ViT) built entirely from scratch using PyTorch.
No pre-trained weights or external transformer libraries were used.

The objective is to classify paddy leaf diseases using real farm images captured under natural conditions.
The final model achieves over 93% validation accuracy and is published on Hugging Face for public use.

Traditional CNNs rely on local receptive fields, whereas Vision Transformers divide images into patches and learn global relationships across the entire image. This enables improved robustness and higher accuracy, especially on diverse datasets.

🌱 Dataset Overview

The dataset contains 10 classes:

9 disease categories

1 healthy leaf category

Images were captured in real farming environments with variations in lighting, angle, clarity, and resolution.

Class Imbalance

The dataset is highly imbalanced:

337 images — Bacterial Panicle Blight (smallest class)

1,764 images — Healthy (largest class)

Blast, Hispa, Dead Heart — each with 1,000+ images

Several minority diseases with only a few hundred samples

This imbalance required careful sampling strategies during training.

🖼️ Data Transformations
Training Transformations

Horizontal flips

Random rotations

Color jitter (brightness & contrast)

Random cropping & scaling

Normalization

These augmentations simulate real-world variations and improve generalization.

Validation Transformations

Resize to 224 × 224

Normalize

No augmentations

This ensures stable and fair model evaluation.

🔀 Train–Validation Split & Class Imbalance Handling

An 80–20 split was created by random shuffling, ensuring proportional representation across all classes.

To handle class imbalance:

A WeightedRandomSampler was used

Minority classes were oversampled

Majority classes were undersampled

This ensures the model sees a balanced set of examples throughout training.

⚙️ Model Configuration

The Vision Transformer is intentionally compact to make training feasible on a single GPU.

Key configuration details:

Image size: 224 × 224

Patch size: 16 × 16

Total patches: 196

Embedding dimension: 256

Transformer blocks: 8

Attention heads: 8

MLP hidden dimension: 512

Classes: 10

Batch size: 64

Learning rate: 3e-4

Epochs: 100

Comparison with ViT-Base/16
Feature	ViT-Base/16	My ViT
Hidden size	768	256
Heads	12	8
Blocks	12	8
Dataset scale	ImageNet/JFT	10-class agricultural dataset

The reduced model is efficient yet powerful enough for this domain-specific dataset.

🔍 How the Vision Transformer Works — Step-by-Step
1. Patch Creation

  The input image (224 × 224 × 3) is divided into 196 patches of size 16 × 16.

2. Patch Embedding

  Each patch is projected into a 256-dimensional embedding.
  Positional embeddings are added so the model can understand patch order.

3. CLS Token

  A learnable CLS token is prepended.
  After processing, this token acts as the final image representation.

4. Transformer Encoder Blocks

  Each block contains:

  LayerNorm
  
  Multi-head self-attention
  
  Skip connections
  
  MLP with GELU activation
  
  These allow the model to learn global relationships across patches.

5. CLS Token Extraction

  After all transformer blocks, the final CLS token is extracted as the image embedding.

6. Classification

  A linear layer converts the CLS token into 10 class logits.

🏋️ Training Strategy

The training pipeline includes:

Batch processing of images

Forward pass to compute predictions

Cross-entropy loss for error calculation

Backpropagation to compute gradients

Adam optimizer for weight updates

Tracking loss and accuracy per epoch

Weighted sampling ensures that minority classes are learned effectively.
Validation is performed on clean, untouched images to evaluate generalization.

📈 Model Performance

93%+ validation accuracy

Smooth decreasing loss curve

Strong generalization

No significant overfitting

Good minority-class performance due to weighted sampling

🤗 Hugging Face Model Publishing

The final trained model is publicly available on Hugging Face.

Repository includes:

pytorch_model.bin (model weights)

config.json (model architecture configuration)

README.md (documentation)

Model ID:
prashanth2000/vit-paddy-disease-classifier

Publishing the model makes it accessible for:

Research

Mobile disease-detection apps

Web-based crop monitoring tools

Further fine-tuning or transfer learning
