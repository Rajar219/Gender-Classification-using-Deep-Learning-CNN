# Gender Classification using CNN

A deep learning project that classifies gender from facial images using Convolutional Neural Networks (CNN) with TensorFlow and Keras.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements a binary classification system to predict gender (Male/Female) from facial images. The model is built using a Convolutional Neural Network trained on the UTKFace dataset.

**Key Highlights:**
- Binary classification (Male: 0, Female: 1)
- Image size: 128x128 pixels
- Training accuracy: ~87%
- Validation accuracy: ~89%

---

## Features

- [x] Automated dataset download from Kaggle
- [x] Data preprocessing and augmentation
- [x] CNN model with 3 convolutional layers
- [x] Early stopping to prevent overfitting
- [x] Model saving in H5 format
- [x] Image prediction with uploaded files
- [x] Visual display of predicted images

---

## Dataset

**UTKFace Dataset**
- Source: Kaggle (jangedoo/utkface-new)
- Total images: 23,708
- Training set: 18,966 images (80%)
- Validation set: 4,742 images (20%)

The dataset contains facial images labeled with age, gender, and ethnicity. This project uses the gender labels for binary classification.

---

## Model Architecture

```
Input Layer: 128x128x3 (RGB Images)
    |
Conv2D (32 filters, 3x3) + ReLU
    |
MaxPooling2D (2x2)
    |
Conv2D (64 filters, 3x3) + ReLU
    |
MaxPooling2D (2x2)
    |
Conv2D (128 filters, 3x3) + ReLU
    |
MaxPooling2D (2x2)
    |
Flatten
    |
Dense (64 units) + ReLU
    |
Dense (1 unit) + Sigmoid
    |
Output: Gender Prediction (0 or 1)
```

**Total Parameters:** 1,699,009 (6.48 MB)

---

## Installation

### Prerequisites

- Python 3.7+
- Google Colab (recommended) or local Jupyter environment
- Kaggle API credentials

### Setup Instructions

1. **Clone or download the notebook:**
   ```bash
   git clone <repository-url>
   cd gender-classification
   ```

2. **Install required packages:**
   ```bash
   pip install tensorflow keras scikit-learn pandas numpy
   pip install keras-tuner kaggle
   ```

3. **Setup Kaggle API:**
   - Download `kaggle.json` from your Kaggle account
   - Place it in `~/.kaggle/` directory
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

---

## Usage

### Training the Model

1. **Upload Kaggle credentials:**
   Run the first cell and upload your `kaggle.json` file

2. **Download and prepare dataset:**
   The notebook automatically downloads and extracts the UTKFace dataset

3. **Train the model:**
   ```python
   # Model trains for 30 epochs with early stopping
   history = model.fit(
       train_generator,
       epochs=30,
       validation_data=validation_generator,
       callbacks=[early_stopping]
   )
   ```

4. **Save the model:**
   ```python
   model.save("final_gender_model.h5")
   ```

### Making Predictions

**Option 1: Without Image Display**
```python
# Load model
model = tf.keras.models.load_model("final_gender_model.h5")

# Upload and predict
uploaded = files.upload()
prediction = model.predict(img_array_preprocessed)
```

**Option 2: With Image Display**
```python
# Display uploaded image and prediction
display(Image(filename=img_path, width=200))
prediction = model.predict(img_array_preprocessed)
```

---

## Results

### Training Performance

| Metric | Training | Validation |
|--------|----------|------------|
| Accuracy | 87.2% | 89.4% |
| Loss | 0.288 | 0.238 |
| Epochs | 30 | 30 |

### Model Performance Over Time

- **Initial Accuracy:** 62.6% (Epoch 1)
- **Final Accuracy:** 89.4% (Epoch 30)
- **Best Validation Loss:** 0.238

The model showed consistent improvement with proper convergence and minimal overfitting.

---

## Project Structure

```
gender-classification/
|
|-- Gender_Classification.ipynb    # Main notebook
|-- final_gender_model.h5          # Trained model (generated)
|-- utkface_dataset/               # Dataset folder (generated)
|   |-- UTKFace/                   # Image files
|-- kaggle.json                    # Kaggle API credentials
|-- README.md                      # Project documentation
```

---

## Requirements

```txt
tensorflow>=2.10.0
keras>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
Pillow>=8.0.0
kaggle>=1.5.0
keras-tuner>=1.1.0
```

---

## Data Augmentation Techniques

The model uses the following augmentation for better generalization:

- Rotation: ±20 degrees
- Width shift: 20%
- Height shift: 20%
- Shear range: 20%
- Zoom range: 20%
- Horizontal flip: Enabled
- Rescaling: 1/255

---

## Model Training Details

**Optimizer:** Adam
**Loss Function:** Binary Crossentropy
**Metrics:** Accuracy
**Batch Size:** 32
**Image Size:** 128x128
**Early Stopping:** Patience of 5 epochs

---

## Future Improvements

- [ ] Implement transfer learning (VGG16, ResNet50)
- [ ] Add age and ethnicity classification
- [ ] Deploy model as web application
- [ ] Add confusion matrix and ROC curve analysis
- [ ] Implement real-time video prediction
- [ ] Optimize model for mobile deployment

---

## Troubleshooting

**Issue:** Kaggle API authentication fails
- Solution: Verify `kaggle.json` is in the correct location with proper permissions

**Issue:** Out of memory during training
- Solution: Reduce batch size or use Google Colab with GPU

**Issue:** Low prediction accuracy
- Solution: Ensure image is preprocessed correctly (128x128, normalized)

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- UTKFace dataset creators and maintainers
- TensorFlow and Keras development teams
- Kaggle for hosting the dataset
- Google Colab for providing free GPU resources

---

## Author

**Raja R**

AI and Python Developer | India

**Project Status:** Active Development

**Last Updated:** January 2026
