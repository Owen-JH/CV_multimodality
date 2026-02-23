import pandas as pd
from sklearn.impute import KNNImputer
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
file_path = r'C:\大学\大四\大四下\毕设\数据集\data.xlsx'
data = pd.read_excel(file_path)

# 计算每列缺失值的数量
missing_values = data.isnull().sum()

# 删除缺失值大于 500 的列
data_cleaned = data.loc[:, missing_values <= 500]

# 删除没有做超声检查的患者样本
data_filtered = data_cleaned[data_cleaned['US_Performed'] != 'no']

# 删除没有编号的患者样本
data_filtered = data_filtered[data_filtered['US_Number'].notna()]
data_filtered = data_filtered.drop(columns=['Diagnosis_Presumptive', 'US_Performed', 'Stool', 'Alvarado_Score', 'Paedriatic_Appendicitis_Score', 'Management', 'Severity'])

# 识别二分类特征
binary_columns = [
    col for col in data_filtered.columns
    if data_filtered[col].dropna().nunique() == 2
]

# 将二分类特征转换为 0 和 1
data_encoded = data_filtered.copy()

# 对每个二分类列进行编码，同时保留 NaN
for col in binary_columns:
    unique_vals = data_encoded[col].dropna().unique()
    if len(unique_vals) == 2:
        val_map = {val: idx for idx, val in enumerate(sorted(unique_vals))}
        data_encoded[col] = data_encoded[col].map(val_map)

binary_columns = data_encoded.columns[data_encoded.nunique() == 2]

# 所有二分类特征的缺失值采用众数填补
for column in binary_columns:
    mode_value = data_encoded[column].mode()[0]
    data_encoded[column].fillna(mode_value, inplace=True)

# 分类特征'Peritonitis' 中的缺失值填补为'no'
data_encoded['Peritonitis'].fillna('no', inplace=True)

# 筛选数值型数据
numerical_columns = data_encoded.select_dtypes(include=['float64', 'int64']).columns
numerical_columns_excluding_binary = [col for col in numerical_columns if col not in binary_columns]

# 排除 'Appendix_Diameter' 这一特征（因为缺失值较多，使用均值填补可能出现误差）
numerical_columns_excluding_binary_and_appendix = [col for col in numerical_columns_excluding_binary if col != 'Appendix_Diameter']

# 使用均值填补其它数值型数据的缺失值
for column in numerical_columns_excluding_binary_and_appendix:
    mean_value = data_encoded[column].mean()
    data_encoded[column].fillna(mean_value, inplace=True)

# Select the columns to apply KNN imputation (excluding non-numeric columns)
numeric_columns = data_encoded.select_dtypes(include=['float64', 'int64']).columns

# 使用KNN填补特征 'Appendix_Diameter' 中的缺失值
knn_imputer = KNNImputer(n_neighbors=5)
data_encoded[numeric_columns] = knn_imputer.fit_transform(data_encoded[numeric_columns])

# Define the mappings for each column
ketones_mapping = {'no': 0, '+': 2, '++': 4, '+++': 6}
rbc_mapping = {'no': 0, '+': 10, '++': 25, '+++': 50}
wbc_mapping = {'no': 0, '+': 15, '++': 50, '+++': 100}

# Apply the mappings to the respective columns
data_encoded['Ketones_in_Urine'] = data_encoded['Ketones_in_Urine'].map(ketones_mapping)
data_encoded['RBC_in_Urine'] = data_encoded['RBC_in_Urine'].map(rbc_mapping)
data_encoded['WBC_in_Urine'] = data_encoded['WBC_in_Urine'].map(wbc_mapping)

# 选择包含缺失值的列
columns_to_impute = ['Ketones_in_Urine', 'RBC_in_Urine', 'WBC_in_Urine']

# 初始化KNN填补器
imputer = KNNImputer(n_neighbors=5)

# 填补缺失值
data_encoded[columns_to_impute] = imputer.fit_transform(data_encoded[columns_to_impute])

# 使用独热编码处理'Peritonitis' 特征
df_encoded = pd.get_dummies(data_encoded, columns=['Peritonitis'])

# 将独热编码后生成的新的二分类特征转换为数值
df_encoded[['Peritonitis_generalized', 'Peritonitis_local', 'Peritonitis_no']] = data_encoded[['Peritonitis_generalized', 'Peritonitis_local', 'Peritonitis_no']].replace({True: 1, False: 0})

# Separate features (X) and target (y)
X = df_encoded.drop(columns=['Diagnosis', 'US_Number', 'Coughing_Pain', 'Appendix_on_US', 'Sex', 'Ipsilateral_Rebound_Tenderness', 'BMI', 'Length_of_Stay', 'Migratory_Pain', 'Thrombocyte_Count', 'Dysuria', 'Peritonitis_local', 'Peritonitis_no', 'Neutrophilia', 'Neutrophil_Percentage', 'WBC_Count', 'Age', 'Height', 'Weight'])  # Exclude Diagnosis and US_Number from features
y = df_encoded['Diagnosis']

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

