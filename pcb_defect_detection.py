# 1. 安裝 TensorFlow 相關
# pip install tensorflow opencv-python matplotlib
# 標籤解析與轉換：YOLO 格式標籤需要轉換為模型所需的格式，並在訓練過程中動態加載。
# 數據增強：可以對訓練集進行數據增強，如旋轉、翻轉、縮放等，以提升模型的泛化能力。
# 模型微調：根據具體應用需求，可以選擇預訓練的 YOLO 模型，並在 DsPCBSD+ 數據集上進行微調。
# 超參數調整：模型的批量大小、學習率、訓練步數等超參數對訓練效果有較大影響。

# 加載 DsPCBSD+ 數據集

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory

# 設定數據集的路徑
dataset_dir = 'path_to_your_dataset/Data_YOLO/images'

# 1. 加載訓練集和驗證集
train_ds = image_dataset_from_directory(
    os.path.join(dataset_dir, 'train'),
    label_mode=None,  # YOLO格式的標籤單獨處理
    image_size=(224, 224),  # 設定圖像尺寸
    batch_size=32,
    shuffle=True
)

val_ds = image_dataset_from_directory(
    os.path.join(dataset_dir, 'val'),
    label_mode=None,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# 2. YOLO 標籤處理 (包括邊界框的坐標以及對應的類別)

# 處理 YOLO 格式的標籤 (每個標籤文件包含類別, 邊界框中心x, y 以及寬度和高度)
def parse_yolo_label(label_path, img_width, img_height):
    boxes = []
    with open(label_path) as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.split())
            # 轉換 YOLO 坐標格式為 [xmin, ymin, xmax, ymax]
            xmin = (x_center - width / 2) * img_width
            ymin = (y_center - height / 2) * img_height
            xmax = (x_center + width / 2) * img_width
            ymax = (y_center + height / 2) * img_height
            boxes.append([xmin, ymin, xmax, ymax, int(class_id)])
    return np.array(boxes)

# # 解析標籤並與圖像對應
# label_path = 'path_to_your_dataset/Data_YOLO/labels/train/example.txt'
# img_width, img_height = 224, 224
# boxes = parse_yolo_label(label_path, img_width, img_height)
# print(boxes)

# 加載圖像和標籤
def load_image_and_label(image_path, label_path):
    """讀取圖像並解析對應的 YOLO 格式標籤"""
    # 加載圖像
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [226, 226])  # 使用原始的 226x226 尺寸

    # 加載標籤
    img_width, img_height = 226, 226
    boxes = parse_yolo_label(label_path, img_width, img_height)

    return img, boxes



# # 3. 建立 YOLO 模型架構

# # 安裝 tensorflow-yolov4-tflite 包
# # pip install tensorflow-yolov4-tflite

# from yolov4.tf import YOLOv4

# # 設置 YOLOv4 模型
# yolo = YOLOv4()

# # 加載預訓練模型權重
# yolo.classes = "path_to_yolo_classes.txt"
# yolo.make_model()
# yolo.load_weights("path_to_pretrained_yolo_weights.weights", weights_type="yolo")

# # 編譯模型
# yolo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # 訓練模型
# history = yolo.fit(train_ds, epochs=20, validation_data=val_ds)

# # 保存訓練好的模型
# yolo.save_weights("yolov4_pcb_weights.h5")

# # 4. 訓練與驗證

# # 訓練模型
# history = yolo.fit(
#     train_ds,
#     epochs=50,               # 訓練 50 個 epoch
#     validation_data=val_ds,   # 驗證集
#     batch_size=32             # 每批次32張圖像
# )

# # 查看模型性能
# yolo.evaluate(val_ds)


# # 5. 模型預測與評估

# import cv2
# import matplotlib.pyplot as plt

# # 加載測試圖像
# img_path = 'path_to_your_dataset/test/image.jpg'
# image = cv2.imread(img_path)
# image = cv2.resize(image, (224, 224))

# # 預測缺陷
# boxes, scores, classes, nums = yolo.predict(image)

# # 可視化結果
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# for i in range(nums[0]):
#     box = boxes[0][i]
#     plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
#                                       fill=False, color='red', linewidth=2))
# plt.show()