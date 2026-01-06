# ❤️ Heart Disease Prediction System
An **end-to-end Machine Learning project** focused on **data analysis, model building, evaluation, and deployment** to predict the risk of heart disease using clinical parameters.
This project demonstrates the **complete ML pipeline**, from raw data to a deployed web application.

---
**Objective:**
To build a **machine learning–based system** that can **predict heart disease risk early** using patient health data, helping in timely medical decision-making.

---
## 🎯 Project Objectives

* Perform detailed **Exploratory Data Analysis (EDA)**
* Preprocess and prepare medical data
* Train and compare multiple ML models
* Tune hyperparameters for best performance
* Select a **reliable final model**
* Deploy the system using **Streamlit**

---
## 📊 Dataset Information

* **Dataset:** UCI Heart Disease Dataset
* **Target Variable:** Presence or absence of heart disease
* **Features Used (13)
---
## 🔄 Complete Machine Learning Pipeline

### 1️⃣ Data Collection
* Loaded clinical heart disease dataset

### 2️⃣ Data Preprocessing

* Handled missing values
* Encoded categorical variables
* Scaled numerical features using **MinMaxScaler**
* Prepared clean dataset for modeling

### 3️⃣ Exploratory Data Analysis (EDA)

* Feature distribution analysis
* Correlation analysis
* Identification of important predictors
* Insights into risk factors

### 4️⃣ Train-Test Split

* Split data into training and testing sets
* Ensured unbiased evaluation

### 5️⃣ Model Training

Multiple models were trained:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Random Forest

### 6️⃣ Hyperparameter Tuning

* Used **RandomizedSearchCV**
* Optimized parameters for Random Forest

### 7️⃣ Model Evaluation

Evaluated models using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

### 8️⃣ Final Model Selection

* **Random Forest** selected as final model
* Performed best on **Recall and ROC-AUC**, which are critical in medical diagnosis

### 9️⃣ Deployment

* Built an interactive **Streamlit web app**
* Includes:

  * Prediction system
  * Data analysis section
  * Model explanation
---
## 📈 Model Performance Summary

* **Final Model:** Tuned Random Forest Classifier
* **Why Random Forest?**
  * Handles non-linear relationships
  * Robust to noise
  * Reduces overfitting using ensemble learning

## 🖥️ Web Application Features

* User-friendly interface
* Patient data input using sliders & dropdowns
* Real-time heart disease risk prediction
* Data analysis and visual insights
* Clean and professional UI

---

## 📁 Project Folder Structure

```bash
Heart_disease_project/
│
├── data/
│   ├── heart_disease.csv
│   └── heart_disease_cleaned.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── models/
│   ├── final_model.pkl
│   └── scaler.pkl
│
├── ui/
│   ├── app.py
│   └── assets/
│       └── heart.png
│
├── requirements.txt
└── README.md
```

---
## ⚙️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Streamlit
* Joblib
---
## ▶️ How to Run the Project

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run ui/app.py
```
---
## ⚠️ Disclaimer

This project is intended **only for educational and research purposes**.
---
## 🚀 Future Improvements

* Larger and more diverse datasets
* Explainable AI (SHAP, LIME)
* Deep learning models
* Clinical validation

---
## ⭐ Conclusion

This project showcases:

* Practical **EDA skills**
* A complete **machine learning pipeline**
* Proper **model evaluation for medical data**
* **Real-world deployment experience**
---

Just say the word — you’ve DONE the hard part 👑
