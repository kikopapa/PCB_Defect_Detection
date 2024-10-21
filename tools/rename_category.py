import os

# 設定標註資料夾路徑
labels_dir = r"D:\KIKOPAPA\my_DsPCBSD+\Data_YOLO\labels"  # 替換為你的 labels 資料夾路徑

# 檢查標註文件的子目錄 (train 和 val)
for subdir in ['train', 'val']:
    subdir_path = os.path.join(labels_dir, subdir)

    # 遍歷該子目錄下所有的標註文件
    for filename in os.listdir(subdir_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(subdir_path, filename)
            
            # 讀取標註文件
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # 修改類別索引 8 為 6
            new_lines = []
            for line in lines:
                values = line.strip().split()
                class_id = int(values[0])

                # 如果是第 8 類，改為第 6 類
                if class_id == 8:
                    values[0] = '6'
                new_lines.append(' '.join(values) + '\n')

            # 將修改後的內容寫回原標註文件
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            
            print(f"Processed {filename} in {subdir} directory")

print("Finished updating class 8 to class 6 in all label files.")
