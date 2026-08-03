# 🔢 Handwritten Digit Recognition (Machine Learning)

Classifying handwritten digits (0–9) using **Logistic Regression** and the **scikit-learn Digits dataset**. This project demonstrates how a machine learning classification algorithm can recognize handwritten numbers and predict custom digit images.


## Overview

This project contains two classification tasks:

 - Multiclass Classification (digits 0–9)
 - Binary Classification (digits 0 and 1)

This project trains a **Logistic Regression** model to recognize handwritten digits from the built-in **Digits dataset** provided by scikit-learn.

After training, the model is evaluated on a test set and then used to predict **custom handwritten digit images** stored in a local folder.

The project demonstrates the complete machine learning workflow, including data loading, model training, evaluation, and prediction on unseen images.


## Features

🔹 Multiclass Digit Classification (0–9)
* 🤖 Logistic Regression classifier
* 📊 Uses the scikit-learn Digits dataset
* 📈 Train/test data splitting
* 🎯 Classification accuracy evaluation
* 📉 Mean Squared Error (MSE) calculation
* 🖼️ Prediction on custom handwritten digit images
* 📚 Beginner-friendly implementation
  
🔹 Binary Digit Classification (0 vs 1)
* Filters the dataset to only digits 0 and 1
* Trains a Logistic Regression classifier
* Compares predicted and actual labels
* Calculates classification accuracy

## Technologies Used

* Python 3
* NumPy
* OpenCV
* Scikit-learn
* Matplotlib


## Dataset

The project uses the **Digits Dataset** from **scikit-learn**.

Dataset characteristics:

* **1,797 samples**
* **64 features per sample (8×8 grayscale image)**
* **10 classes (digits 0–9)**

Each image is flattened into a one-dimensional feature vector before being used for training.


## Machine Learning Workflow

1. Load the Digits dataset.
2. Split the data into training and testing sets.
3. Train a Logistic Regression classifier.
4. Evaluate the trained model.
5. Predict custom handwritten digit images.
6. Measure prediction accuracy.


## Project Structure

```text
Digits-Classification-Logistic-Regression/
│
├── digits_logistic_regression.py
├── binary_digit_classifier.py
├── test dataset/
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── ...
│   └── 9.jpg
├── requirements.txt
├── LICENSE
└── README.md
```


## Installation

Clone the repository:

```bash
git clone https://github.com/Matin-python/Digits-Classification-Logistic-Regression.git
```

Move into the project directory:

```bash
cd Digits-Classification-Logistic-Regression
```

Install the required packages:

```bash
pip install -r requirements.txt
```

or install them manually:

```bash
pip install numpy opencv-python scikit-learn matplotlib
```


## How to Run

Multiclass Classification
```bash
python digits_logistic_regression.py
```
Binary Classification
```bash
python binary_digit_classifier.py
```

The program will:

* Load the Digits dataset.
* Train the Logistic Regression model.
* Evaluate its performance.
* Predict the digits inside the **test dataset** folder.
* Display prediction accuracy.


## Evaluation Metrics

The project reports:

* ✅ Classification Accuracy
* 📉 Mean Squared Error (MSE)

These metrics evaluate the model's performance on the test dataset.


## Example Output

```text
==================================================
digits.data.shape = (1797, 64)

digits.images.shape = (1797, 8, 8)
==================================================

Correct Prediction = 97.5%

Mean Squared Error = 0.04

Predicted Output = [0 1 2 3 4 5 6 7 8 9]

Real Output = [0 1 2 3 4 5 6 7 8 9]

Accuracy Score = 100%
```

*(Results may vary because the training/testing data is randomly split each run.)*


## Related Projects

This project is part of a collection exploring different machine learning algorithms using the **Digits dataset**.

### 🎯 Handwritten Digit Clustering using K-Means

An unsupervised learning project that groups handwritten digits into clusters using the K-Means algorithm without using class labels during training.

➡️ **Repository:** *Add my K-Means repository link here.*


## Future Improvements

* 📊 Confusion matrix visualization
* 📈 Classification report (Precision, Recall, F1-score)
* 🎨 GUI for drawing digits
* 📷 Real-time webcam digit recognition
* 🧠 Compare multiple classification algorithms
* 🔄 Hyperparameter tuning
* 💾 Save and load trained models
* 🌐 Deploy as a web application


## Contributing

Contributions, suggestions, and bug reports are welcome. Feel free to fork the repository and submit a pull request.


## License

This project is licensed under the MIT License.


## Author

**Mohammad Reza Bakhshandeh**

Electrical Engineering (Electronics) Graduate

Interested in Python Development, Machine Learning, Deep Learning, Computer Vision, Artificial Intelligence, and Game Development.
