# Cat vs Dog Image Classification Using CNN

## Project Overview

This project implements a Convolutional Neural Network (CNN) to classify images into two categories: **Cat** and **Dog**.

The model was developed and trained using **TensorFlow/Keras** in Google Colab. Images are resized to `64 × 64` pixels and processed using image rescaling and data augmentation before being passed to the CNN.

## Objective

The objective of this project is to build a basic image classification model that can:

* Learn visual features from cat and dog images.
* Classify an input image as either **Cat** or **Dog**.
* Make predictions on new, unseen images.

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Google Colab

## Dataset Structure

The dataset is organized into separate training and testing directories:

```text
cat_dog_data/
│
├── training_set/
│   └── training_set/
│       ├── cats/
│       └── dogs/
│
└── test_set/
    └── test_set/
        ├── cats/
        └── dogs/
```

The `cats` and `dogs` folders are used as the two classification classes.



## Image Preprocessing

The images are resized to:

```text
64 × 64 × 3
```

where:

* `64 × 64` = image height and width
* `3` = RGB color channels

For training images, the following preprocessing and augmentation techniques were used:

* Rescaling pixel values using `1./255`
* Shear transformation
* Zoom transformation
* Horizontal flipping

For test images, only pixel rescaling was applied.

## CNN Architecture

The CNN consists of the following layers:

```text
Input Image (64 × 64 × 3)
        ↓
Conv2D (32 filters, 3 × 3)
        ↓
ReLU Activation
        ↓
MaxPooling2D (2 × 2)
        ↓
Conv2D (32 filters, 3 × 3)
        ↓
ReLU Activation
        ↓
MaxPooling2D (2 × 2)
        ↓
Flatten
        ↓
Dense (128 neurons, ReLU)
        ↓
Dense (1 neuron, Sigmoid)
        ↓
Cat / Dog
```

### Model Components

**Conv2D**

Extracts visual features such as edges, textures, and patterns from the images.

**MaxPooling2D**

Reduces the spatial dimensions of the feature maps while retaining important features.

**Flatten**

Converts the 2D feature maps into a one-dimensional vector.

**Dense Layer**

Learns higher-level patterns from the extracted features.

**Sigmoid Output Layer**

Produces a value between `0` and `1` for binary classification.

The class mapping used in the project is:

```text
cats → 0
dogs → 1
```

Therefore:

```text
Prediction ≤ 0.5 → Cat
Prediction > 0.5 → Dog
```

## Model Training

The model was trained using:

```python
history = classifier.fit(
    training_set,
    validation_data=test_set,
    epochs=10
)
```

The training process uses batches of images and evaluates the model on the test data during training.

## Results

The trained CNN achieved approximately:

**87% accuracy**

The exact result can vary depending on the dataset, preprocessing, and training run.

## New Image Prediction

After training, the model can be used to classify a new image.

The input image is:

1. Loaded.
2. Resized to `64 × 64`.
3. Converted into a NumPy array.
4. Normalized to the range `0–1`.
5. Passed to the trained CNN.

Example prediction:

```text
Prediction value: 0.38741
Prediction: Cat
```

Since the prediction value was below `0.5`, the model classified the image as **Cat**.

## Project Workflow

```text
Dataset
   ↓
Image Preprocessing
   ↓
Data Augmentation
   ↓
CNN Model
   ↓
Model Training
   ↓
Model Evaluation
   ↓
New Image Prediction
```

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

Open:

```text
CNN_Cat_Dog_Classification.ipynb
```

The notebook can be run using Google Colab or a local Jupyter environment with the required dependencies.


## Future Improvements

Possible improvements to this project include:

* Using a larger or more diverse dataset.
* Adding more CNN layers.
* Experimenting with different filter sizes and numbers of filters.
* Using Dropout to reduce overfitting.
* Using Transfer Learning with models such as VGG16, ResNet, or MobileNet.
* Adding a separate validation dataset.
* Building a simple web interface for image prediction.

## Author

**Priya Tudu**

`
