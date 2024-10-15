# args.yaml
- 內容：這是模型訓練過程中使用的超參數和配置文件，記錄了訓練時所使用的所有參數，例如 batch size、學習率、數據集路徑等。
- 用途：方便重現訓練過程，或者進行參數調整。

# weights/best.pt 跟 weights/last.pt
- 內容：該目錄包含訓練過程中生成的權重文件（如 best.pt 和 last.pt）。
- 用途：保存了最佳模型和最後一次訓練的模型，這些權重可以用於進行推理或進一步的調整。