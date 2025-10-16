
## 🛰️ GPS Fault Detection & Integrity Monitoring using ML and Neural Networks

### 📘 Overview

This project focuses on **fault detection and integrity monitoring** in satellite navigation systems (GPS). It combines **classical RAIM (Receiver Autonomous Integrity Monitoring)** techniques with **machine learning classifiers** such as **SVM, Random Forest, XGBoost, and Neural Networks** to improve fault classification and decision-making reliability.

The workflow involves simulating satellite data, generating training and testing datasets, and evaluating various classification models for detecting faulty satellites in real-time.

---

## 🧩 Project Structure

| File                     | Description                                                                                                                                                                                          |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **constants.py**         | Contains key GPS and simulation constants used across other modules.                                                                                                                                 |
| **gen_data.py**          | Generates the train and test datasets using simulation parameters. <br>Uncomment regions within the script based on your data generation needs (e.g., number of satellites, fault types).            |
| **gps_im_classifier.py** | Parent class defining the structure for data generation and classifier training. It integrates components like the **State Estimator**, **Integrity Monitor (IM)**, **Classifier**, and **Sampler**. |
| **naebt.py**             | Main implementation class for the system — integrates **Extended Kalman Filter (EKF)**, **Integrity Monitor**, **Sampler**, and **SVM** modules.                                                     |
| **RAIM.py**              | Executes the RAIM algorithm for classical fault detection and performance comparison against ML-based methods.                                                                                       |
| **snaekf.py**            | Implements a **Sampler** assuming a Gaussian prior on received measurements. Used for generating noisy satellite data samples.                                                                       |
| **svm.py**               | Script for training and testing the **SVM classifier** on the generated dataset.                                                                                                                     |
| **SVMtest.py**           | Used to validate and tune **SVM hyperparameters** (kernel, C, gamma, etc.).                                                                                                                          |
| **SVM_to_RAIM.py**       | Loads a trained **SVM model** and compares its performance on single-fault satellite datasets against **RAIM** results.                                                                              |
| **train_env.py**         | Simulates satellite and agent trajectories based on environment parameters defined in `gen_data.py`. Used to generate realistic GPS measurement sequences.                                           |
| **util.py**              | Contains **utility functions** and the **core RAIM algorithm implementation** used by multiple modules.                                                                                              |

---

## 🤖 Neural Network and ML Integration (Colab)

The **Neural Network**, **Random Forest**, and **XGBoost** implementations are available in the following Google Colab notebook:

🔗 [Neural Network Colab](https://colab.research.google.com/drive/1pLewKTTS9Xe2bAenxXHcK5bzQkLSZcLb?usp=sharing)

This notebook performs:

* Data loading and preprocessing (from the generated GPS data files)
* Training and evaluation of:

  * **1-layer Neural Network**
  * **2-layer Neural Network**
  * **Random Forest**
  * **XGBoost**
  * *(Optional)* **SVM (RBF)** for baseline comparison
* Accuracy computation and visualization through bar plots

Example result output:

```
Model Performance Comparison
----------------------------
SVM (RBF):       0.921
Neural Net:      0.937
Random Forest:   0.955
XGBoost:         0.961
```

---

## ⚙️ How to Run the Project

### 🧠 1. Generate Training and Testing Data

```bash
python gen_data.py
```

➡️ Make sure to **uncomment** the relevant sections in `gen_data.py` depending on your simulation needs (e.g., single or multiple fault cases).

### 🧭 2. Run Classical RAIM

```bash
python RAIM.py
```

Generates baseline results for GPS fault detection using the RAIM algorithm.

### 💻 3. Train and Test the SVM

```bash
python svm.py
```

To inspect SVM parameters or fine-tune them:

```bash
python SVMtest.py
```

### 🧩 4. Compare SVM vs RAIM

```bash
python SVM_to_RAIM.py
```

Compares trained SVM predictions with classical RAIM results for a dataset containing single satellite faults.

### 🧠 5. Neural Network / ML Evaluation (Google Colab)

Open the linked Colab notebook, upload your generated train/test data, and run all cells to train:

* Neural Networks (1-layer & 2-layer)
* Random Forest
* XGBoost
* SVM (optional baseline)

---

## 📂 Data Folder Notes

> ⚠️ Always ensure the generated data files (`train_data_*.csv`, `test_data_*.csv`, etc.) are **copied to the correct folder** before running other scripts such as `svm.py` or `RAIM.py`.

---

## 📈 Outputs

Each model (RAIM, SVM, NN, RF, XGB) generates:

* Classification accuracy
* Confusion matrix
* Precision, recall, and F1-score
* Saved result text files (e.g., `NN_1layer_6nodes_test.txt`, `RandomForest_results.txt`)

---

## 🧮 Dependencies

Install the required Python packages:

```bash
pip install numpy pandas scikit-learn xgboost matplotlib
```

For Neural Network experiments (Colab):

```bash
pip install tensorflow torch torchvision
```

---

## 📊 Visualization Example

The Colab notebook automatically plots performance comparison among models:

```python
Model Performance Comparison
─────────────────────────────
SVM (RBF):       0.921
Neural Net:      0.937
Random Forest:   0.955
XGBoost:         0.961
```

<p align="center">
  <img src="https://github.com/placeholder/gps-project-demo/blob/main/results/model_comparison.png" width="550">
</p>

---

## 🧠 Authors

**Aritra Ghosh**, [Team Name / Research Group]
PES University, Bengaluru

---

## 🧩 Summary

This project combines traditional **integrity monitoring (RAIM)** with modern **machine learning and neural network techniques** to enhance satellite fault detection reliability.
It bridges classical signal integrity approaches and modern data-driven fault classification.


