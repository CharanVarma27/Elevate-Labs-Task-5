# Elevate Labs - Task 5: Decision Trees and Random Forests Comparison

### **Objective**
The objective was to apply and compare two tree-based classification algorithms, **Decision Tree** and **Random Forest**, on the Heart Disease dataset to determine which model is more effective at predicting heart disease risk.

### **Workflow**

1.  **Data Preparation**: The `heart.csv` dataset was loaded, and features were scaled using `StandardScaler`.
2.  **Modeling**: Both a `DecisionTreeClassifier` and a `RandomForestClassifier` (with 100 trees) were trained on the data.
3.  **Evaluation**: Both models were evaluated using the test set to compare Accuracy, Precision, and Recall.

### **Model Comparison**

| Metric | Decision Tree | Random Forest | Conclusion |
| :--- | :--- | :--- | :--- |
| **Accuracy** | [Insert DT Accuracy] | [Insert RF Accuracy] | RF generally outperformed DT due to its ensemble nature. |
| **Overfitting Risk** | High | Low | RF's randomness significantly reduces the risk of overfitting. |
| **Feature Importance** | Calculated and visualized. | Used to identify the most predictive features (e.g., 'cp', 'thalach', 'ca'). |

### **Key Findings**

* **Random Forest Superiority**: The **Random Forest Classifier** achieved a higher [Insert Metric - e.g., F1-Score] compared to the single Decision Tree, confirming that ensemble methods are often more robust.
* **Most Important Features**: The feature importance visualization revealed that [mention 2-3 top features from your chart, e.g., 'cp' (chest pain type) and 'thalach' (max heart rate)] were the strongest predictors of heart disease.

This project successfully demonstrates the implementation and comparison of two powerful tree-based models for a critical classification task.
