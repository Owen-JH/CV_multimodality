import pandas as pd

# 读取 CSV 文件，确保处理无误
csv_path = 'patient_features_ResNet18.csv'
df = pd.read_csv(csv_path)

# 提取患者编号的第一个数字作为分组 ID
df['Group_ID'] = df['Patient_ID'].astype(str).str.split('.').str[0]

# 确保 Group_ID 是字符串格式
df['Group_ID'] = df['Group_ID'].astype(str)

# 获取特征列（所有以 'Feature_' 开头的列）
feature_columns = [col for col in df.columns if col.startswith('Feature_')]

# 均值融合
fused_df_mean = df.groupby('Group_ID')[feature_columns].mean().reset_index()

# 中位数融合
fused_df_median = df.groupby('Group_ID')[feature_columns].median().reset_index()

# 保存融合结果
fused_df_mean.to_csv(r'C:\cat\patient_features_fusion_mean.csv', index=False)
fused_df_median.to_csv(r'C:\cat\patient_features_fusion_median.csv', index=False)



print("特征融合完成，结果已保存。")
