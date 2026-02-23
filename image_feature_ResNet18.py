import os
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights
import numpy as np
import pandas as pd

# 设置图像文件夹路径
image_folder = r'C:\Crop'  # 请修改为你的图像文件夹路径
save_path = r'C:\cat\patient_features_ResNet18.csv'  # 最终特征保存路径

# 定义预处理和标准化
preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载预训练的 ResNet 模型（去掉最后的全连接层）
resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

# 组织患者图像
patient_images = {}
image_files = os.listdir(image_folder)

for image_file in image_files:
    if not image_file.endswith(('.jpg', '.png', '.bmp')):
        continue

    patient_id = image_file.split('_')[0]

    if patient_id not in patient_images:
        patient_images[patient_id] = []

    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Failed to load image: {image_file}")
        continue

    # 图像预处理
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = preprocess(image_rgb).unsqueeze(0)

    with torch.no_grad():
        feature = resnet(image_tensor).squeeze().numpy()
        patient_images[patient_id].append(feature)

# 特征融合（平均）并组织为CSV格式
patient_features = []

for patient_id, features in patient_images.items():
    if len(features) > 0:
        # 将特征堆叠并计算平均
        features_array = np.vstack(features)
        fused_feature = np.mean(features_array, axis=0)
        patient_features.append([patient_id] + fused_feature.tolist())
        print(f"✅ 患者 {patient_id} - 图像数：{len(features)} - 特征维度：{fused_feature.shape}")
    else:
        print(f"⚠️ 患者 {patient_id} 没有有效图像，跳过。")

# 将特征保存为CSV文件
columns = ['Patient_ID'] + [f'Feature_{i+1}' for i in range(512)]
df = pd.DataFrame(patient_features, columns=columns)
df.to_csv(save_path, index=False)
print(f"✅ 所有患者的图像特征已提取并保存到 {save_path}")
