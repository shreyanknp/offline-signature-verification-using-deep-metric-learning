# Offline Signature Verification using Deep Metric Learning
Automated Offline Handwritten Signature Verification and Forgery Detection using Siamese Neural Network

This project implements an **offline handwritten signature verification system** using **deep metric learning and Siamese neural networks**. The objective is to determine whether two signature images belong to the same writer by learning a similarity-based embedding space.

Instead of treating the problem as a traditional classification task, the model learns feature embeddings for signature images and compares them using a distance metric. Genuine signature pairs produce smaller distances, while forged pairs produce larger distances.

---

## Datasets Used

The models were trained and evaluated using three publicly available benchmark datasets:

- **CEDAR Dataset**
- **BHSig260 Dataset**
- **GPDS150 Dataset**

These datasets contain both **genuine signatures and skilled forgeries**, which makes them suitable for evaluating signature verification systems.

---

## Models Implemented

Several deep learning architectures were explored as backbone feature extractors in the Siamese framework:

- **Baseline CNN**
- **MobileNetV2**
- **ResNet50**

Each model generates **signature embeddings**, and the similarity between two signatures is computed using **Euclidean distance**.

---

## Evaluation Metrics

Model performance was evaluated using standard biometric verification metrics:

- **False Acceptance Rate (FAR)**
- **False Rejection Rate (FRR)**
- **Equal Error Rate (EER)**

Cross-dataset experiments were also conducted to analyze the **generalization capability of the models**.

---

## Key Features

- Writer-independent signature verification  
- Siamese neural network architecture  
- Multiple CNN backbone models  
- Cross-dataset evaluation across multiple datasets  
- Experiments on real-world benchmark datasets  

---

## Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Matplotlib  

---

## Author

**Shreyank N P**  
M.Sc. Applied Data Science and Analytics  
SRH University Heidelberg