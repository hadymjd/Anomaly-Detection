# 💳 Unsupervised Fraud Detection with One-Class SVM

## Project Overview

This project focuses on building a robust anomaly detection system for a highly imbalanced credit card transaction dataset. The goal is to accurately identify rare fraudulent transactions (anomalies) using unsupervised machine learning techniques, specifically when no labeled fraud data is available for training.

The primary objective was to maximize **Recall** (catching fraud) while maintaining acceptable **Precision** (minimizing false alarms).

-----

## 📊 Key Results Summary

| **Model** | **Precision (Accuracy of Flags)** | **Recall (Fraud Caught)** | **F1-Score** |
| :--- | :--- | :--- | :--- |
| **One-Class SVM (Winner)** | 39.07% | **81.71%** | 0.5286 |
| **Isolation Forest** | 61.54% | 27.64% | 0.3815 |

**Conclusion:** The **One-Class SVM** model was selected for deployment due to its superior **Recall ($\sim 82\%$)**. In financial fraud detection, the cost of missing a fraud case (False Negative) is much higher than the operational cost of reviewing a false alarm (False Positive).

-----

## 🛠 Methodology and Pipeline

### 1\. Data Preparation and Feature Engineering

  * **Imbalance Handling:** The dataset was split to ensure the training data was **purely normal ($\text{Class}=0$)**, which is mandatory for unsupervised anomaly detection models.
  * **Feature Engineering:** The original `Time` feature was transformed into an **`Hour` of Day** feature to capture the bimodal cyclical pattern of transactions.
  * **Scaling:**
      * `Amount` was scaled using **RobustScaler** to mitigate the impact of extreme outliers.
      * The `Hour` feature was scaled using **StandardScaler**.
      * The PCA-transformed `V1`-`V28` features were used directly.

### 2\. Model Selection and Training

Two core unsupervised models were benchmarked:

  * **Isolation Forest (iForest):** An ensemble tree-based method that isolates anomalies by randomly partitioning data.
  * **One-Class SVM (OCSVM):** A kernel-based method that finds a non-linear boundary (hyperplane) to enclose the normal data, classifying everything outside that boundary as an anomaly.
  * **Contamination Factor:** Both models were trained using a contamination factor ($\nu$) set to the true fraud rate of `0.001727` to tune their sensitivity.

### 3\. Anomaly Characterization (Insights)

The analysis of transactions flagged by the OCSVM model revealed key patterns that can inform investigative teams:

  * **Amount:** Flagged anomalies generally have a **significantly higher scaled transaction amount** than predicted normal transactions.
  * **V-Feature Signals:** The model flags transactions deviating heavily in specific PCA dimensions, indicating that the fraud signal is not localized to one factor but is a **structural deviation** in the high-dimensional space.
  * **Time:** The engineered `Hour` feature helps the model identify transactions occurring during low-activity periods (e.g., late night/early morning) as higher risk.

-----

## ⚙️ How to Run the Project

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/Anomaly-Detection.git
    cd Anomaly-Detection
    ```

2.  **Install Dependencies:**

    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

3.  **Data Requirement:** Due to the file size limit, the `creditcard.csv` file (143MB) is managed using Git LFS. Ensure Git LFS is installed:

    ```bash
    git lfs install
    ```

4.  **Run Pipeline:** Execute the sequential Python scripts:

      * **Data Splitting & Preprocessing:** (Code from steps 1-3)
      * **Model Training & Evaluation:** (Code from step 4)
      * **Anomaly Characterization:** (Code from step 5)

*Created by Hadi Majed*

-----
