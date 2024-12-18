## Note: 
- 數據集名稱：DsPCBSD+。
- 數據來源：數據集中的圖像來自於實際生產的 PCB 板，透過 AOI（自動光學檢查）設備拍攝。
- 圖像數量：10,259 張。
- 標記數量：20,276 個手動標記 (缺陷邊界框)。
- 缺陷類別：9 種（短路、尖刺、雜銅、斷路、鼠咬、孔崩、導體刮痕、導體異物、基材異物）。
- 訓練/驗證集比例：8:2。
- 驗證模型：Co-DETR 和 YOLOv6-L6，mAP 均超過 0.84
  - mAP（mean Average Precision）
  - 召回率（Recall）

----

## 1. 背景介紹

- 印刷電路板（PCB）在製造過程中經常會出現多種表面缺陷，這些缺陷不僅影響外觀，還可能對電路板的性能造成損害。
因此，檢測 PCB 表面缺陷對於品質管控至關重要。
- 傳統的缺陷檢測方式主要依賴人工視覺檢查，存在主觀性強、效率低下等問題。
- 深度學習技術的快速發展提供了更高效且精準的缺陷檢測方案，然而，這需要大量且多樣的數據集來進行模型訓練。
- 該研究建立了一個包含 9 種 PCB 表面缺陷類別的數據集（DsPCBSD+），這些缺陷根據其成因、位置和形態進行分類，總計收集了 10,259 張圖像，並手動註解了 20,276 個缺陷 ( 一張圖像可能包含多種缺陷 )。
該數據集旨在推動基於深度學習的 PCB 表面缺陷檢測研究。

## 2. 方法

- 缺陷圖像收集：來自實際生產的內層和外層 PCB 的缺陷圖像，通過自動光學檢測（AOI）設備收集，圖像經過預處理，去除了噪音，並增強了對比度和亮度，最終生成 32,259 張圖片，圖像大小為 226 x 226 像素。

- 缺陷分類與數據預處理：基於缺陷的成因（如 銅殘留 Copper residue、銅不足 Copper deficiency、導體刮痕 Conductor scratch 和 異物 Foreign object）進行分類，並進一步細分為 9 個類別：

  - 短路 (Short) - SH
  - 尖刺 (Spur) - SP
  - 雜銅 (Spurious copper) - SC
  - 斷路 (Open) - OP
  - 鼠咬 (Mouse bite) - MB
  - 孔崩 (Hole breakout) - HB
  - 導體刮痕 (Conductor scratch) - CS
  - 導體異物 (Conductor foreign object) - CFO
  - 基材異物 (Base material foreign object) - BMFO

  ![image](doc/PCB_surface_defect_classification.png)

- 缺陷標記與數據集劃分：利用 LabelImg 工具對每個缺陷進行手動邊界框註解，過濾掉無缺陷的圖像、重複缺陷圖像和不完整的缺陷圖像。最終生成包含 20,276 個缺陷標記的 10,259 張圖像。數據集被劃分為訓練集和驗證集，比例為 8:2。

## 3. 數據集的技術驗證

研究選用了兩種 SOTA 模型（Co-DETR 和 YOLOv6-L6）進行驗證。訓練過程中對超參數進行了調整，如批量大小和學習率等。驗證結果顯示，這些模型在檢測 PCB 表面缺陷時表現出色，平均精度（mAP）均高於 0.84，顯示出該數據集對於深度學習模型的實用性和可靠性。
結果顯示，這些模型在多數缺陷類別上都有較高的檢測準確性，特別是在處理大範圍缺陷時表現良好，但對於小型缺陷的檢測仍有挑戰。


## 4. 數據集的使用說明

數據集提供了 YOLO 和 COCO 格式，存儲了訓練和驗證數據的圖像和標籤文件。
標籤文件包含了缺陷的類別、邊界框的坐標及其尺寸，這些數據可直接用於訓練深度學習模型。

  ![image](doc/Count_of_three_size_labels.png)
  ![image](doc/distribution_of_defect_labels_in_dataset_across_different_categories.png)

以下使用 Python 搭配 Seaborn 套件進行資料視覺化繪圖
  ![image](tools/Figure_v5.png)
> tools > visualization.py


## 5. 數據集的優勢與局限性

- 該數據集克服了以往數據集的一些不足，例如缺陷樣本不足、標籤不均衡等問題，為基於深度學習的 PCB 缺陷檢測研究提供了有力支持。
- 該數據集的局限性在於，它僅包含 2D 圖像，無法檢測如凸起或凹陷等三維缺陷。
- 影像來自於 PCB 的內外層，未包含到焊接遮蔽後的缺陷圖像
- 所有圖像皆為局部區域的裁剪，未整合成整板圖像，這在實際應用中可能需要額外處理來定位缺陷

### Ref : A Dataset for Deep Learning-Based Detection of Printed Circuit Board Surface Defect
