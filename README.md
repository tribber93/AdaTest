# Trash Image Classification using Pre-trained Model Vision Transformer (ViT)

This repository contains an implementation of an image classification model using a pre-trained Vision Transformer (ViT) model from Hugging Face. The model is fine-tuned to classify images into six categories: cardboard, glass, metal, paper, plastic, and trash.

## Dataset

The dataset consists of images from six categories from [`garythung/trashnet`](https://huggingface.co/datasets/garythung/trashnet) with the following distribution:

- Cardboard: 806 images
- Glass: 1002 images
- Metal: 820 images
- Paper: 1188 images
- Plastic: 964 images
- Trash: 274 images

Due to the imbalance in the dataset, i use a `WeightedRandomSampler` to handle class imbalance during training.

## Model

We utilize the pre-trained Vision Transformer model [`google/vit-base-patch16-224-in21k`](https://huggingface.co/google/vit-base-patch16-224-in21k) from Hugging Face for image classification. The model is fine-tuned on the dataset to achieve optimal performance.

The trained model is accessible on Hugging Face Hub at: [`tribber93/my-trash-classification`](https://huggingface.co/tribber93/my-trash-classification)

## Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Training

To train the model, use the provided Jupyter Notebook:

1. [`Image_Classification_Test_Yoni_Tribber.ipynb`](Image_Classification_Test_Yoni_Tribber.ipynb): Contains the training pipeline, including data preprocessing, augmentation, and evaluation.

## Results

After training, the model achieved the following performance:

| Epoch | Training Loss | Validation Loss | Accuracy |
|-------|---------------|-----------------|----------|
| 1     | 3.3200        | 0.7011          | 86.25%   |
| 2     | 1.6611        | 0.4298          | 91.49%   |
| 3     | 1.4353        | 0.3563          | 94.26%   |

### Key Features:
- Data augmentation.
- Weighted sampling to address class imbalance.
- Model fine-tuning using Pre-trained Model Vision Transformer (ViT).