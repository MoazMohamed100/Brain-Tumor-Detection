# Brain Tumor Detection Using CNN

## Overview

This project focuses on detecting brain tumors from MRI images using a Convolutional Neural Network (CNN). The trained deep learning model classifies MRI scans into two categories:

* **Yes**: Brain tumor detected
* **No**: No brain tumor detected

A user-friendly **Streamlit web application** is provided to allow users to upload MRI images and obtain real-time predictions along with confidence scores.

---

## Project Structure

```
├── Brain_Tumor_Detection.ipynb   # Model training and evaluation notebook
├── App.py                       # Streamlit web application
├── brain_tumor_cnn.keras        # Trained CNN model
├── README.md                    # Project documentation
```

---

## Dataset

* MRI brain images
* Two classes: **Tumor** and **No Tumor**
* Images are resized to **224 × 224** pixels
* Pixel values are normalized to the range **[0, 1]**

> The dataset is preprocessed and split into training and validation sets inside the notebook.

---

## Model Architecture

* Convolutional Neural Network (CNN)
* Key components:

  * Convolutional layers
  * Max-pooling layers
  * Fully connected (Dense) layers
  * Softmax output layer

The model is trained using **TensorFlow / Keras** and saved in `.keras` format.

---

## Technologies Used

* **Python 3**
* **TensorFlow / Keras**
* **NumPy**
* **Pillow (PIL)**
* **Streamlit**

---

## Deployment Using Streamlit

The trained CNN model has been deployed as an **interactive Streamlit web application**, allowing users to perform brain tumor detection directly through a browser interface.

The application enables users to:

* Upload MRI images (`.jpg`, `.jpeg`, `.png`)
* View the uploaded image
* Receive a prediction (**Yes / No**) indicating tumor presence
* See the model's confidence score

---

## How to Run the Project

### 1. Install Dependencies

```bash
pip install tensorflow streamlit numpy pillow
```

### 2. Run the Streamlit Application

```bash
streamlit run App.py
```

### 3. Use the Application

1. Upload an MRI image (`.jpg`, `.jpeg`, or `.png`)
2. The image will be displayed
3. The model predicts whether a tumor is present
4. Prediction confidence is shown

---

## Application Output

* **Prediction label**: Yes / No
* **Confidence score**: Probability percentage of the predicted class

---

## Example Use Cases

* Educational demonstrations of medical image classification
* Assisting in preliminary analysis of MRI scans
* Deep learning project for healthcare applications

---

## Limitations

* The model is not intended for clinical diagnosis
* Performance depends on dataset quality and diversity
* Works only with MRI images similar to the training data

---

## Future Improvements

* Multi-class tumor classification
* Model optimization and fine-tuning
* Deployment on cloud platforms
* Integration with medical imaging systems

---

## Author

**Moaz Mohamed**

---

## Disclaimer

This project is for **educational purposes only** and should not be used as a substitute for professional medical diagnosis.
