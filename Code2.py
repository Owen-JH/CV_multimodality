import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 打印混淆矩阵
def print_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix for {model_name}:")
    print(cm)
    print()


# 绘制 ROC 曲线的函数
def plot_roc_curve(model, X_test_scaled, y_test, model_name, color):
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob = model.decision_function(X_test_scaled)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})', color=color)

# 读取Excel文件
file_path = 'data2.xlsx'
df = pd.read_excel(file_path)

# Separate features (X) and target (y)
X = df.drop(columns=['Diagnosis', 'US_Number', 'Group_ID', 'Coughing_Pain', 'Appendix_on_US', 'Sex', 'Ipsilateral_Rebound_Tenderness', 'BMI', 'Length_of_Stay', 'Migratory_Pain', 'Thrombocyte_Count', 'Dysuria', 'Peritonitis_local', 'Peritonitis_no', 'Neutrophilia', 'Neutrophil_Percentage', 'WBC_Count', 'Age', 'Height', 'Weight'])  # Exclude Diagnosis and US_Number from features
y = df['Diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred = log_reg.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display the evaluation metrics
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

print(metrics)
# Logistic Regression 混淆矩阵
print_confusion_matrix(y_test, y_pred, "Logistic Regression")

# Initialize and train Support Vector Machine (SVM) model
svm_model = SVC(kernel='linear')  # Using linear kernel
svm_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_svm = svm_model.predict(X_test_scaled)

# Evaluate the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

# Display the evaluation metrics for SVM
metrics_svm = {
    "Accuracy": accuracy_svm,
    "Precision": precision_svm,
    "Recall": recall_svm,
    "F1 Score": f1_svm
}

print(metrics_svm)
# SVM 混淆矩阵
print_confusion_matrix(y_test, y_pred_svm, "SVM")

# Initialize and train Random Forest Classifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

# Display the evaluation metrics for Random Forest
metrics_rf = {
    "Accuracy": accuracy_rf,
    "Precision": precision_rf,
    "Recall": recall_rf,
    "F1 Score": f1_rf
}

print(metrics_rf)
# Random Forest 混淆矩阵
print_confusion_matrix(y_test, y_pred_rf, "Random Forest")

plt.figure(figsize=(10, 7))
plot_roc_curve(log_reg, X_test_scaled, y_test, "Logistic Regression", 'blue')
plot_roc_curve(svm_model, X_test_scaled, y_test, "SVM", 'green')
plot_roc_curve(rf_model, X_test_scaled, y_test, "Random Forest", 'red')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()


