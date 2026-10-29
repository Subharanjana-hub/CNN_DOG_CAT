# CNN - Dog and Cat Image Classification

A Convolutional Neural Network with three convolutional layers, built with TensorFlow and Keras, that classifies images as either a dog or a cat.

## About the Project

This is a binary image classification project. The network reads 150 x 150 color photos and decides whether each one shows a cat or a dog. Three convolutional layers learn features step by step, from simple edges to more complex shapes, and a final sigmoid neuron gives the probability that the image is a dog.

## Features

- Automatic download of the cats and dogs dataset (2000 training and 1000 validation images)
- Option to use your own images from a `dataset/` folder
- Data augmentation (random flip, rotation and zoom) to reduce overfitting
- Exactly three convolutional layers, each followed by max pooling
- Dropout before the classifier to reduce overfitting
- Callbacks: EarlyStopping, ReduceLROnPlateau and ModelCheckpoint
- Training curves for accuracy and loss
- Evaluation with a classification report, confusion matrix and ROC curve
- Sample predictions with confidence scores
- A `predict_image()` function to test your own photo

## Model Architecture

| Layer | Details |
|-------|---------|
| Input | 150 x 150 x 3 image |
| Augmentation | RandomFlip, RandomRotation, RandomZoom |
| Rescaling | Pixel values scaled to 0-1 |
| Conv Layer 1 | Conv2D (32 filters, 3x3, ReLU) + MaxPooling |
| Conv Layer 2 | Conv2D (64 filters, 3x3, ReLU) + MaxPooling |
| Conv Layer 3 | Conv2D (128 filters, 3x3, ReLU) + MaxPooling |
| Classifier | Flatten, Dropout 0.5, Dense (128, ReLU) |
| Output | Dense (1), sigmoid (cat = 0, dog = 1) |
| Loss | Binary cross-entropy |
| Optimizer | Adam (learning rate 0.001) |

## Requirements

- Python 3.9 or higher
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

Install everything with:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Dataset

By default, the script downloads the cats and dogs dataset the first time you run it (about 65 MB, so an internet connection is needed).

To use your own images, create a folder named `dataset` next to the script with this structure:

```
dataset/
    train/
        cat/
        dog/
    validation/
        cat/
        dog/
```

## Usage

```bash
python cnn_dog_cat.py
```

Training is much faster on a GPU. On a CPU, reduce the `EPOCHS` value at the top of the script.

To classify your own photo, add this line at the end of the script:

```python
predict_image("my_pet.jpg")
```

## Output

All results are saved in the `cnn_dog_cat_outputs/` folder:

| File | Description |
|------|-------------|
| `samples.png` | Sample training images |
| `training_curves.png` | Accuracy and loss over epochs |
| `confusion_matrix.png` | Confusion matrix on the validation set |
| `roc_curve.png` | ROC curve with AUC score |
| `predictions.png` | Sample predictions with confidence |
| `best_cnn.keras` | Best model during training |
| `cnn_dog_cat.keras` | Final trained model |

## Results

Add your validation accuracy and AUC here after running the script.

