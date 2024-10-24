import os

# 設定資料集的路徑
dataset_path = r"D:\KIKOPAPA\my_DsPCBSD+\Data_YOLO"  # 替換為你的資料集路徑
images_path = os.path.join(dataset_path, "images","train")  # 圖像目錄  # train 跟 val
labels_path = os.path.join(dataset_path, "labels","train")  # 標註目錄  # train 跟 val

# 設定要移除的類別
remove_classes = {6, 7}  # 要移除的類別索引

# 檢查圖像和標註檔
for filename in os.listdir(labels_path):
    if filename.endswith(".txt"):
        # 讀取標註檔
        annotation_file_path = os.path.join(labels_path, filename)
        with open(annotation_file_path, 'r') as f:
            lines = f.readlines()

        # 過濾不需要的類別
        new_lines = []
        for line in lines:
            values = list(map(float, line.strip().split()))
            class_id = int(values[0])
            if class_id not in remove_classes:
                new_lines.append(line)

        # 如果有過濾，則寫入新檔案
        if new_lines:
            with open(annotation_file_path, 'w') as f:
                f.writelines(new_lines)
        else:
            # 如果所有類別都被移除，可以選擇刪除該檔案
            os.remove(annotation_file_path)
            print(f"Removed empty annotation file: {annotation_file_path}")

            # 刪除對應的圖片檔案
            image_file_name = os.path.splitext(filename)[0] + '.jpg'  # 假設圖檔為 JPG 格式
            image_file_path = os.path.join(images_path, image_file_name)
            if os.path.exists(image_file_path):
                os.remove(image_file_path)
                print(f"Removed corresponding image file: {image_file_path}")

print("Finished processing annotations in labels/train.")
