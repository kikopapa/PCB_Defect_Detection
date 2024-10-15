## results.png
- 內容：包含訓練過程中的多種評估指標，如 Precision、Recall、mAP (mean Average Precision) 在不同 epoch 的變化。
- 用途：幫助觀察訓練和驗證的性能隨著 epoch 的變化，了解模型的收斂情況
  
## confusion_matrix.png

## val_batchX_labels.jpg 跟 val_batchX_pred.jpg
- 內容：這些圖像展示了驗證數據集上的模型預測結果 (val_batchX_pred.jpg) 和真實標籤 (val_batchX_labels.jpg)，X 表示批次號。
- 用途：用來可視化驗證數據的預測效果，了解模型是否準確地識別了目標物件。

## train_batchX.jpg
- 內容：這些圖像展示了訓練批次中輸入數據和對應標籤的可視化結果，X 表示批次號。
- 用途：用來檢查數據加載過程中的標註是否正確，了解數據在進行增強處理後的狀況

## labels.jpg
- 內容：顯示數據集中所有標籤的分佈情況。
- 用途：用來檢查數據集中各類別的數量分佈，是否存在數據不平衡問題。

## labels_correlogram.jpg
- 內容：顯示數據集中標籤之間的相關性（co-occurrence），用來檢查標籤是否經常一起出現。
- 用途：用來觀察不同類別之間的相關性，可能有助於理解數據集的結構

## F1_curve.png
- 顯示模型在不同信心閾值下的F1分數曲線

## R_curve.png
- 召回率(Recall)曲線，顯示在不同信心閾值下的召回率變化

## args.yaml
- 內容：這是模型訓練過程中使用的超參數和配置文件，記錄了訓練時所使用的所有參數，例如 batch size、學習率、數據集路徑等。
- 用途：方便重現訓練過程，或者進行參數調整。
  
## weights/best.pt 跟 weights/last.pt
- 內容：該目錄包含訓練過程中生成的權重文件（如 best.pt 和 last.pt）。
- 用途：保存了最佳模型和最後一次訓練的模型，這些權重可以用於進行推理或進一步的調整。
