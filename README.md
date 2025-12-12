## Monocular Depth Estimation (EfficientNetB0 & Transfer Learning)

This project contains the complete code for training a Deep Learning model to perform Monocular Depth Estimation (predicting depth maps from single RGB images). The solution leverages Transfer Learning using a pre-trained EfficientNetB0 backbone to achieve efficient and accurate depth regression.

## 🚀 Key Features

The pipeline is engineered for efficiency and accuracy within a Kaggle environment, utilizing modern Convolutional Neural Network (CNN) architectures and post-processing refinement.

* **Transfer Learning**: Utilizes EfficientNetB0 (pretrained on ImageNet) as a powerful feature extractor, allowing the model to understand complex visual features with fewer training epochs.

* **Regression-Based Decoding**: Instead of a traditional U-Net decoder, this approach uses a dense regression head to predict depth values directly from global features.

* **Custom Metrics**: Implements a custom Root Mean Squared Error (RMSE) metric to rigorously track model performance during training.

* **Post-Processing Refinement**: Includes a Gaussian smoothing step to reduce noise and artifacts in the final depth map predictions.

* **Robust Training Loop**: Features dynamic learning rate adjustment (ReduceLROnPlateau) and EarlyStopping to prevent overfitting.

## 📈 Methodology
The core of this solution is mapping high-level semantic features from RGB images to continuous depth values using a frozen encoder and a trainable regression head.

### 1. Data Preprocessing & Generation

To handle memory constraints and ensure consistent input formats, a custom data_generator is implemented:

* **Resizing**: All input RGB images and target depth maps are resized to 128x128.

* **Normalization**: Pixel values are scaled to the range [0, 1].

* **Streaming**: Data is loaded in batches on-the-fly to maintain low RAM usage during training.

### 2. Model Architecture (Transfer Learning)

The model consists of two distinct stages:

* **Encoder (Backbone)**: EfficientNetB0 (weights=imagenet). This section is frozen (trainable=False) to preserve learned feature representations.

* **Decoder (Head)**: A series of fully connected layers designed to map features to depth pixels:

* **GlobalAveragePooling2D** to flatten spatial dimensions.

* **Dense layers (512 -> 256 neurons)** with ReLU activation for non-linearity.

* A final Dense layer with 128×128 neurons to predict every pixel's depth.

* Reshape layer to reconstruct the (128, 128, 1) spatial output.

### 3. Training & Inference

* **Optimizer**: Adam with a learning rate of 1e-4.

* **Loss Function**: Mean Squared Error (MSE).

* **Refinement**: During inference, the raw predictions are passed through a scipy.ndimage.gaussian_filter (sigma=1). This smooths out pixel-jitter and results in more visually coherent depth maps.

* **Output Generation**: The script generates visual depth maps (saved as images) and a structured CSV file for competition submission.

### 🛠️ Tech Stack

* Core: Python 3
* TensorFlow, Keras (Functional API)
* OpenCV (cv2)
* EfficientNetB0
* Data Handling: Pandas, NumPy
* Post-Processing: Scipy (Gaussian Filter)

### 🏃 Running the Project

### 1. Dependencies

This script is designed to run in a Kaggle Notebook or a standard Python environment with the following libraries installed:

```
pip install tensorflow opencv-python pandas numpy scipy
```

### 2. Dataset

This model was trained on a depth estimation dataset as part of a university challenge. The data consists of pairs of RGB images and their corresponding ground-truth depth maps. Due to privacy and access restrictions, the dataset is not publicly available and is not included in this repository.

Therefore, the script cannot be run out-of-the-box without downloading the specific competition data separately and placing it in the correct directory structure (e.g., training/images and training/depths).

### 3. Notebook Review

The provided code serves as an end-to-end pipeline:

* **Model Build**: Initializes EfficientNet and attaches the custom regression head.

* **Training**: Fits the model using generators and saves the best weights (model.keras).

* **Inference**: Loads the best model, predicts on test data, applies smoothing, and saves results.


