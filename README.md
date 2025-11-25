Vision Transformer (ViT) for Paddy Leaf Disease Classification

A from-scratch PyTorch implementation trained on real agricultural images

Overview

This project presents a fully custom implementation of a Vision Transformer (ViT) built entirely from scratch using PyTorch. No pre-trained weights or external transformer libraries are used. The goal of the model is to classify paddy leaf diseases using real farm images captured under natural conditions. The final model achieves more than 93% validation accuracy and is published on Hugging Face for public use and experimentation.

Traditional CNNs focus on local patterns using convolutional filters, whereas Vision Transformers divide an image into patches and learn global relationships across the entire image. This allows the model to capture long-range dependencies and significantly improves robustness and accuracy, especially on diverse datasets.

Dataset Overview

The dataset consists of real photographs of paddy leaves belonging to ten categories. Nine categories represent various diseases, and one corresponds to healthy leaves. Images were captured in real farming environments, meaning lighting, angle, background, and resolution vary significantly.

The dataset is highly imbalanced. Some examples include:

The smallest class (Bacterial Panicle Blight) has 337 images

The largest class (Healthy) has 1,764 images

Classes like Blast, Hispa, and Dead Heart contain over 1,000 images

Several minority diseases have only a few hundred samples

This imbalance required a training strategy that ensures fair learning across all classes.

Data Transformations

All images were resized to 224 × 224 pixels to maintain a consistent input format for the Vision Transformer.

Training set transformations included:

Horizontal flips

Random rotations

Color jitter for brightness and contrast

Random cropping and scaling

Normalization

These augmentations simulate real-world inconsistencies experienced in farm conditions, improving the model’s generalization.

Validation set transformations included:

Only resizing and normalization

No augmentations

This ensures evaluation is always performed on clean and consistent images.

Train–Validation Split and Class Imbalance Handling

An 80–20 split was used, created by randomly shuffling all image indices to ensure each class was proportionally represented.

Because the dataset is imbalanced, a weighted sampling strategy was implemented during training. Classes with fewer images were oversampled, while majority classes were undersampled. This ensures the model receives balanced training signals without modifying or duplicating images in the dataset.

Model Configuration

The Vision Transformer implemented here is intentionally compact, balancing complexity with practical training needs.

Key configuration details:

Image size: 224 × 224

Patch size: 16 × 16

Total patches: 196

Embedding dimension (token size): 256

Transformer blocks: 8

Attention heads: 8

MLP hidden dimension: 512

Number of classes: 10

Batch size: 64

Learning rate: 3e-4

Training epochs: 100

Comparison with ViT-Base/16

The original ViT-Base/16 model uses:

Hidden size 768

12 attention heads

12 transformer blocks

Such a large configuration is unnecessary for a 10-class agricultural dataset. The reduced architecture used here trains faster and fits comfortably on a single GPU while still capturing the essential behavior of Vision Transformers.

How the Vision Transformer Works — Step-by-Step
1. Patch Creation

The 224 × 224 × 3 image is divided into 16 × 16 patches. This produces 196 fixed-size image patches arranged in sequence form.

2. Patch Embedding

Each patch is transformed into a 256-dimensional vector representation. Positional embeddings are added so the model knows the spatial location of each patch in the original image.

3. CLS Token

A learnable classification token (CLS token) is added to the beginning of the sequence. After the transformer processes all patches, this token carries the final image-level representation.

4. Transformer Encoder Blocks

Each encoder block contains:

Layer normalization

Multi-head self-attention

Residual connections

A feed-forward MLP layer

These blocks allow patches to interact globally, capturing long-range relationships across the leaf image.

5. CLS Token Extraction

After the sequence passes through all transformer blocks, only the updated CLS token is extracted. It represents the final learned summary of the entire image.

6. Classification

A simple classification head maps the CLS token representation into 10 output logits corresponding to the disease categories.

Training Strategy

The training follows a standard deep-learning workflow:

Images are processed in batches

Forward pass computes predictions

Cross-entropy loss measures the prediction error

Backpropagation calculates gradients

The Adam optimizer updates model weights

Accuracy is tracked during training

Loss and accuracy for each epoch are recorded

Weighted sampling ensures balanced learning despite dataset imbalance. Validation is performed at the end of each epoch on untouched, clean images to measure generalization.

Model Performance

Final validation accuracy: 93%+

Smooth reduction in training loss across epochs

Strong generalization due to augmentations and weighted sampling

No significant overfitting observed

Hugging Face Model Publishing

The final trained model was published on the Hugging Face Hub for easy reuse. The repository includes:

Model weights (pytorch_model.bin)

Architecture configuration (config.json)

Project documentation (README.md)

The model is available under the ID:

prashanth2000/vit-paddy-disease-classifier

Publishing the model allows anyone to load, test, extend, or integrate it into real-world applications such as mobile disease-diagnosis systems or API-based crop monitoring tools.
