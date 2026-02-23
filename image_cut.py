import os
from PIL import Image

# 文件夹路径
folder_path = 'C:\大学\大四\大四下\毕设\数据集\pic\p'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 拼接完整路径
    file_path = os.path.join(folder_path, filename)

    # 仅处理图片文件
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
        # 打开图片
        img = Image.open(file_path)
        width, height = img.size

        # 计算中线
        mid = width // 2

        # 左右两半
        left_img = img.crop((0, 0, mid, height))
        right_img = img.crop((mid, 0, width, height))

        # 文件名和扩展名
        name, ext = os.path.splitext(filename)

        # 保存左右两半
        left_img.save(os.path.join(folder_path, f'{name}_left{ext}'))
        right_img.save(os.path.join(folder_path, f'{name}_right{ext}'))

        # 关闭图片
        img.close()

        # 删除原图
        os.remove(file_path)

print("处理完成！")
