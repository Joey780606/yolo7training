# Project Context: Yolo7 Training Program

## Code Specifications

- Programming Language: Python 3.13.9

- Yolo Version: Yolo v7 (Reference: https://github.com/WongKinYiu/yolov7)

- Other Tools: Your choice

- Variable Naming: Use camelCase consistently

- Annotation Language: Traditional Chinese

- Error Handling: All API calls must include a try-catch block

## Program Functionality

- I want to design a program for training with YOLO v7. Since I'm not familiar with it, please write it according to your understanding of standard YOLO training methods.

## Project Progress

### ✅ Completed Development (Phase 1-5)

#### Phase 1: Foundation - 完成
- [x] `requirements.txt` - Python 依賴套件
- [x] `utils/general.py` - 通用工具函式 (座標轉換、NMS、IoU 計算)
- [x] `utils/torchUtils.py` - PyTorch 輔助函式 (裝置選擇、EMA、學習率排程)
- [x] `configs/data/custom.yaml` - 資料集設定範本
- [x] `configs/models/yolov7.yaml` - YOLOv7 標準版架構定義
- [x] `configs/models/yolov7Tiny.yaml` - YOLOv7 輕量版架構定義

#### Phase 2: Model Architecture - 完成
- [x] `models/common.py` - 通用模組 (Conv, Bottleneck, SPPCSPC, ELANBlock, RepConv)
- [x] `models/backbone.py` - E-ELAN 骨幹網路
- [x] `models/neck.py` - FPN + PAN 特徵融合網路
- [x] `models/head.py` - 多尺度檢測頭
- [x] `models/yolo.py` - 完整的 YOLOv7 模型類別

#### Phase 3: Data Pipeline - 完成
- [x] `utils/augmentations.py` - 資料增強 (Mosaic, MixUp, HSV, 翻轉)
- [x] `utils/datasets.py` - 資料集載入和處理
- [x] `scripts/prepareData.py` - 資料格式轉換 (VOC/COCO → YOLO)
- [x] `scripts/splitDataset.py` - 資料集分割工具

#### Phase 4: Training Infrastructure - 完成
- [x] `utils/loss.py` - 損失函數 (CIoU Loss, 物件性, 分類)
- [x] `utils/metrics.py` - 評估指標 (mAP, Precision, Recall, F1)
- [x] `train.py` - 主訓練腳本

#### Phase 5: Inference & Utilities - 完成
- [x] `detect.py` - 推論/檢測腳本
- [x] `validate.py` - 驗證腳本
- [x] `models/__init__.py` - 模型模組初始化
- [x] `utils/__init__.py` - 工具模組初始化

### 📋 Next Production Steps

#### Step 1: 環境設定
```bash
# 安裝 Python 依賴
pip install -r requirements.txt
```

#### Step 2: 資料準備
```bash
# 如果有 VOC 格式標註，轉換為 YOLO 格式
python scripts/prepareData.py --source ./rawData --format voc --output ./data

# 如果有 COCO 格式標註
python scripts/prepareData.py --source ./annotations.json --images ./images --format coco --output ./data

# 分割資料集 (80% 訓練, 20% 驗證)
python scripts/splitDataset.py --source ./data --train 0.8 --val 0.2
```

#### Step 3: 設定資料集
- 編輯 `configs/data/custom.yaml`
- 修改 `train` 和 `val` 路徑指向您的資料
- 修改 `nc` (類別數量) 和 `names` (類別名稱)

#### Step 4: 開始訓練
```bash
# 基本訓練
python train.py --data configs/data/custom.yaml --epochs 100 --batchSize 16

# 使用輕量版模型
python train.py --data configs/data/custom.yaml --model yolov7tiny --epochs 100

# 從預訓練權重繼續訓練
python train.py --data configs/data/custom.yaml --weights weights/yolov7.pt --epochs 50
```

#### Step 5: 驗證模型
```bash
python validate.py --weights runs/train/exp/weights/best.pt --data configs/data/custom.yaml
```

#### Step 6: 執行檢測
```bash
# 檢測圖片
python detect.py --weights runs/train/exp/weights/best.pt --source ./testImages

# 檢測影片
python detect.py --weights runs/train/exp/weights/best.pt --source video.mp4

# 即時攝影機
python detect.py --weights runs/train/exp/weights/best.pt --source 0
```

### 🔄 Optional Future Enhancements

- [ ] 下載並整合官方預訓練權重
- [ ] 新增 TensorBoard 訓練視覺化
- [ ] 新增模型匯出功能 (ONNX, TensorRT)
- [ ] 新增分散式訓練支援
- [ ] 新增自動超參數調整
- [ ] 新增範例資料集 (Sample Dataset)
