import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# 读取Excel文件
file_path = r'C:\大学\大四\大四下\毕设\数据集\data1.xlsx'
data= pd.read_excel(file_path)
scaler = StandardScaler()
# Step 1: Drop 'US_Number' and 'Diagnosis' columns
df_cleaned = data.drop(columns=['US_Number', 'Diagnosis'])

# 归一化
df_scaled = scaler.fit_transform(df_cleaned)

# Apply PCA
pca = PCA()
pca_components = pca.fit_transform(df_scaled)

# 计算方差贡献率及累计贡献率
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# 获取前26个主成分的负载矩阵
loadings = pca.components_[:26]

# 将负载矩阵转换为DataFrame，便于查看
loading_df = pd.DataFrame(loadings.T,
                          columns=[f'PC{i+1}' for i in range(26)],
                          index=df_cleaned.columns)

# 导出前26个主成分的负载矩阵
loading_df.to_excel(r'C:\大学\大四\大四下\毕设\数据集\PCA_Loadings.xlsx', index=True)

# 将前26个主成分数据导出
pca_26_components = pca_components[:, :26]
pca_26_df = pd.DataFrame(pca_26_components, columns=[f'PC{i+1}' for i in range(26)])
pca_26_df.to_excel(r'C:\大学\大四\大四下\毕设\数据集\PCA_Components.xlsx', index=False)

# 统计每个主成分中绝对值最大的前10个特征
top_features = {}

for pc in loading_df.columns:
    # 取该主成分绝对值最大的前10个特征
    top10 = loading_df[pc].abs().nlargest(10).index.tolist()
    for feature in top10:
        top_features[feature] = top_features.get(feature, 0) + 1

# 转换为DataFrame并排序
top_features_df = pd.DataFrame(list(top_features.items()), columns=['Feature', 'Count'])
top_features_df = top_features_df.sort_values(by='Count', ascending=False).reset_index(drop=True)
print(top_features_df)