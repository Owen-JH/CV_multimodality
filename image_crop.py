import os
import cv2
import re

# 图像预处理函数：裁剪为正方形后缩放为 224x224
def preprocess_image(image, target_size=(224, 224)):
    h, w, _ = image.shape
    if w > h:
        top_crop = int(h * 0.10)
        bottom_crop = int(h * 0.10)
        cropped_image = image[top_crop:h - bottom_crop, :]
        new_h, new_w, _ = cropped_image.shape
        start_x = (new_w - new_h) // 2
        final_image = cropped_image[:, start_x:start_x + new_h]
    elif h > w:
        left_crop = int(w * 0.10)
        right_crop = int(w * 0.10)
        cropped_image = image[:, left_crop:w - right_crop]
        new_h, new_w, _ = cropped_image.shape
        start_y = (new_h - new_w) // 2
        final_image = cropped_image[start_y:start_y + new_w, :]
    else:
        final_image = image

    # Resize 到 224x224
    final_image = cv2.resize(final_image, target_size, interpolation=cv2.INTER_AREA)
    return final_image

# 设置图像文件夹路径
image_folder = r'C:\Pictures'  # 原始图像路径
save_folder = r'C:\crop'  # 处理后图像保存路径
os.makedirs(save_folder, exist_ok=True)

image_files = os.listdir(image_folder)
patient_images = {}
file_pattern = re.compile(r'(\d+)\.(\d+)')

for image_file in image_files:
    if not image_file.endswith(('.jpg', '.png', '.bmp')):
        continue

    match = file_pattern.match(image_file)
    if match:
        patient_id = int(match.group(1))
        image_num = match.group(2)

        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"❌ Failed to load image: {image_file}")
            continue

        # 👉 预处理：裁剪 + Resize
        image_processed = preprocess_image(image)

        # 保存处理后的图像
        file_name, ext = os.path.splitext(image_file)
        save_name = f"{file_name}_crop{ext}"
        save_path = os.path.join(save_folder, save_name)
        cv2.imwrite(save_path, image_processed)

        # 添加到患者图像字典
        if patient_id not in patient_images:
            patient_images[patient_id] = []
        patient_images[patient_id].append(image_processed)

        print(f"✅ Patient {patient_id} - Image {image_num} cropped, resized, and saved to {save_name}.")

# 打印统计
total_images = len([f for f in image_files if f.endswith(('.jpg', '.png', '.bmp'))])
print(f"\nTotal original images: {total_images}")
print(f"Total patients: {len(patient_images)}")
