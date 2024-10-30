import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from collections import Counter

# 使用 Seaborn 的預設主題
sns.set_theme(style="whitegrid")

# 設置支持中文的字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 使負號正常顯示

# Load class names from yaml file
def load_class_names(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as file:
        class_data = yaml.safe_load(file)
        class_names = list(class_data['names'].values())
    return class_names

# Function to count samples in each class
def count_samples(labels_path, num_classes):
    class_counts = Counter()
    for label_file in os.listdir(labels_path):
        if label_file.endswith('.txt'):
            try:
                with open(os.path.join(labels_path, label_file), 'r') as f:
                    for line in f:
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1
            except Exception as e:
                print(f"Error reading {label_file}: {e}")

    for i in range(num_classes):
        if i not in class_counts:
            class_counts[i] = 0
    return class_counts

# Set paths for train and val labels (Windows format)
train_labels_path = r'D:\KIKOPAPA\DsPCBSD+\Data_YOLO\labels\train'
val_labels_path = r'D:\KIKOPAPA\DsPCBSD+\Data_YOLO\labels\val'
class_names = load_class_names(r'G:\我的雲端硬碟\my_project\pcb.yaml')

# Total number of classes
num_classes = len(class_names)

# Count samples
train_counts = count_samples(train_labels_path, num_classes)
val_counts = count_samples(val_labels_path, num_classes)

# Convert counts to arrays
train_counts_array = np.array([train_counts[i] for i in range(num_classes)])
val_counts_array = np.array([val_counts[i] for i in range(num_classes)])

# Create x-axis for class labels
x = np.arange(num_classes)

# Plotting with Seaborn
plt.figure(figsize=(12, 6))
# plt.title('Class Images Distribution in Train and Validation Sets', fontsize=16)
plt.title('訓練集與驗證集中的類別影像分佈圖', fontsize=16)

# 使用 Seaborn 的調色板
palette = sns.color_palette("Set2", 2)
bar_train = plt.bar(x - 0.2, train_counts_array, width=0.4, label='Train', color=palette[0], alpha=0.7)
bar_val = plt.bar(x + 0.2, val_counts_array, width=0.4, label='Validation', color=palette[1], alpha=0.7)

# Set x-ticks and labels
plt.xticks(x, class_names, rotation=45, ha='right')
plt.xlabel('缺陷類型', fontsize=12)  # 新增 X 軸標題
plt.ylabel('缺陷影像數量')
plt.legend()

# 顯示數值標記
for i, bar in enumerate(bar_train):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom')

for i, bar in enumerate(bar_val):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom')

# 調整底部邊距
plt.subplots_adjust(bottom=0.2)
                    
plt.show()
