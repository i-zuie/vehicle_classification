import os
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

images_folder = "./dataset/images"  
output_folder = "data"  

train_ratio = 0.7  # 70% 训练集
val_ratio = 0.2    # 20% 验证集
test_ratio = 0.1   # 10% 测试集

classes = ['Bus','Microbus','Minivan','Sedan','SUV','Truck']
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 按类别整理图片并划分数据集
def organize_images(input_folder, output_folder, train_ratio, val_ratio, test_ratio):
    for split in ['train', 'val', 'test']:
        create_dir(os.path.join(output_folder, split))
    category_dict = {}  # 类别: 图片路径列表
    for file in os.listdir(input_folder):
        if file.endswith(('.jpg', '.png', '.jpeg')):  
            file_name = file.split('.')[0]
            category_file = open('dataset/Annotations/%s.xml' % (file_name), encoding='utf-8')
            tree = ET.parse(category_file)
            root = tree.getroot()
            size = root.find('size')
            category_path = os.path.join(input_folder, file)
            if size != None:
                for obj in root.iter('object'):
                    difficult = obj.find('difficult').text
                    category =  obj.find('name').text
                    print(f"{file_name}.xml {category}")
                    if category not in classes or int(difficult) == 1:
                        continue
            if category not in category_dict:
                category_dict[category]=[]
            category_dict[category].append(category_path)
    
    # 划分数据集
    for category, images in category_dict.items():
        print(f"Processing category: {category}, total images: {len(images)}")
        train_val_images, test_images = train_test_split(images, test_size=test_ratio, random_state=42)
        train_images, val_images = train_test_split(train_val_images, test_size=val_ratio/(train_ratio + val_ratio), random_state=42)
        for split, split_images in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
            split_category_folder = os.path.join(output_folder, split, category)
            create_dir(split_category_folder)
            for img_path in split_images:
                shutil.copy(img_path, split_category_folder)
    print("Image organization complete!")


if __name__ == "__main__":
    organize_images(images_folder, output_folder, train_ratio, val_ratio, test_ratio)
