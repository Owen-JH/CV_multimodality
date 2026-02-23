import pandas as pd
# 读取已经融合的患者特征文件和Excel数据
fused_csv_path = 'patient_features_fusion_median.csv'
excel_path = 'data1.xlsx'

# 读取融合后的特征和Excel文件
fused_df = pd.read_csv(fused_csv_path)
excel_df = pd.read_excel(excel_path)

# 确保 US_Number 列为字符串，以便与 Group_ID 匹配
excel_df['US_Number'] = excel_df['US_Number'].astype(str)
fused_df['Group_ID'] = fused_df['Group_ID'].astype(str)

# 将 Group_ID 与 US_Number 进行匹配并合并
merged_df = pd.merge(excel_df, fused_df, left_on='US_Number', right_on='Group_ID', how='inner')

# 保存合并后的数据为新的 Excel 文件
merged_output_path = 'data2.xlsx'
merged_df.to_excel(merged_output_path, index=False)


