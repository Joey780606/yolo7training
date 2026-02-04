"""
通用工具函式模組 (General Utilities Module)

此模組包含 YOLOv7 訓練與推論所需的通用工具函式，包括：
- 檔案與目錄操作
- 座標格式轉換
- 非極大值抑制 (NMS)
- 邊界框處理函式

所有函式都使用 camelCase 命名規則，並包含詳細的繁體中文註解。
"""

import os
import glob
import math
import random
import time
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import torch
import torchvision


def setRandomSeed(seed: int = 42) -> None:
    """
    設定隨機種子以確保實驗可重現性

    為了確保每次訓練的結果都一致，我們需要固定所有隨機數產生器的種子。
    這包括 Python 內建的 random、NumPy 和 PyTorch。

    參數:
        seed: 隨機種子值，預設為 42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 如果使用 CUDA，也要設定 CUDA 的隨機種子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 確保 cuDNN 使用確定性演算法
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def checkFileExists(filePath: str) -> bool:
    """
    檢查檔案是否存在

    參數:
        filePath: 檔案路徑

    返回:
        如果檔案存在返回 True，否則返回 False
    """
    return os.path.isfile(filePath)


def checkDirExists(dirPath: str) -> bool:
    """
    檢查目錄是否存在

    參數:
        dirPath: 目錄路徑

    返回:
        如果目錄存在返回 True，否則返回 False
    """
    return os.path.isdir(dirPath)


def makeDirectory(dirPath: str, exist_ok: bool = True) -> Path:
    """
    建立目錄（如果不存在的話）

    參數:
        dirPath: 要建立的目錄路徑
        exist_ok: 如果目錄已存在是否忽略錯誤

    返回:
        建立的目錄 Path 物件
    """
    path = Path(dirPath)
    path.mkdir(parents=True, exist_ok=exist_ok)
    return path


def incrementPath(basePath: str, existOk: bool = False, sep: str = '') -> Path:
    """
    遞增路徑編號（用於自動命名實驗資料夾）

    例如：runs/train/exp -> runs/train/exp2 -> runs/train/exp3
    這在訓練時很有用，可以自動為每次實驗創建新的資料夾。

    參數:
        basePath: 基礎路徑
        existOk: 如果路徑存在是否返回原路徑
        sep: 數字分隔符

    返回:
        遞增後的路徑
    """
    path = Path(basePath)

    if path.exists() and not existOk:
        # 如果路徑存在，尋找下一個可用的編號
        suffix = path.suffix
        stem = path.stem

        for n in range(2, 9999):
            newPath = path.with_name(f'{stem}{sep}{n}{suffix}')
            if not newPath.exists():
                path = newPath
                break

    path.mkdir(parents=True, exist_ok=True)
    return path


def xyxy2xywh(xyxy: np.ndarray) -> np.ndarray:
    """
    將邊界框從 [x1, y1, x2, y2] 格式轉換為 [x_center, y_center, width, height] 格式

    YOLO 使用的標準格式是中心點座標加上寬高，這種格式在計算損失時更方便。

    座標說明:
    - xyxy 格式: [左上角x, 左上角y, 右下角x, 右下角y]
    - xywh 格式: [中心點x, 中心點y, 寬度, 高度]

    參數:
        xyxy: 形狀為 (N, 4) 的 numpy 陣列，N 是邊界框數量

    返回:
        形狀為 (N, 4) 的 numpy 陣列，使用 xywh 格式
    """
    xywh = np.copy(xyxy)
    # 計算中心點 x = (x1 + x2) / 2
    xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
    # 計算中心點 y = (y1 + y2) / 2
    xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
    # 計算寬度 w = x2 - x1
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    # 計算高度 h = y2 - y1
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return xywh


def xywh2xyxy(xywh: np.ndarray) -> np.ndarray:
    """
    將邊界框從 [x_center, y_center, width, height] 格式轉換為 [x1, y1, x2, y2] 格式

    這是 xyxy2xywh 的逆運算，常用於將 YOLO 格式轉換為其他框架使用的格式。

    參數:
        xywh: 形狀為 (N, 4) 的 numpy 陣列

    返回:
        形狀為 (N, 4) 的 numpy 陣列，使用 xyxy 格式
    """
    xyxy = np.copy(xywh)
    # 計算 x1 = x_center - width/2
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    # 計算 y1 = y_center - height/2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    # 計算 x2 = x_center + width/2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    # 計算 y2 = y_center + height/2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return xyxy


def xyxy2xywhTorch(xyxy: torch.Tensor) -> torch.Tensor:
    """
    將邊界框從 xyxy 格式轉換為 xywh 格式 (PyTorch 版本)

    參數:
        xyxy: 形狀為 (N, 4) 的張量

    返回:
        形狀為 (N, 4) 的張量，使用 xywh 格式
    """
    xywh = xyxy.clone()
    xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2  # 中心點 x
    xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2  # 中心點 y
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]  # 寬度
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]  # 高度
    return xywh


def xywh2xyxyTorch(xywh: torch.Tensor) -> torch.Tensor:
    """
    將邊界框從 xywh 格式轉換為 xyxy 格式 (PyTorch 版本)

    參數:
        xywh: 形狀為 (N, 4) 的張量

    返回:
        形狀為 (N, 4) 的張量，使用 xyxy 格式
    """
    xyxy = xywh.clone()
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # x1
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # y1
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # x2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # y2
    return xyxy


def boxIou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    計算兩組邊界框之間的 IoU (Intersection over Union，交集除以聯集)

    IoU 是衡量兩個邊界框重疊程度的重要指標：
    - IoU = 交集面積 / 聯集面積
    - 範圍在 0 到 1 之間
    - IoU = 1 表示完全重疊
    - IoU = 0 表示完全不重疊

    參數:
        box1: 形狀為 (N, 4) 的張量，格式為 [x1, y1, x2, y2]
        box2: 形狀為 (M, 4) 的張量，格式為 [x1, y1, x2, y2]

    返回:
        形狀為 (N, M) 的 IoU 矩陣
    """
    # 計算每個框的面積
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # 計算交集區域的座標
    # 使用廣播來計算所有配對的交集
    interX1 = torch.max(box1[:, None, 0], box2[:, 0])
    interY1 = torch.max(box1[:, None, 1], box2[:, 1])
    interX2 = torch.min(box1[:, None, 2], box2[:, 2])
    interY2 = torch.min(box1[:, None, 3], box2[:, 3])

    # 計算交集面積（如果沒有交集，則為 0）
    interArea = (interX2 - interX1).clamp(min=0) * (interY2 - interY1).clamp(min=0)

    # 計算 IoU = 交集 / (面積1 + 面積2 - 交集)
    iou = interArea / (area1[:, None] + area2 - interArea + 1e-7)

    return iou


def boxCiou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    計算 CIoU (Complete IoU) 損失

    CIoU 是對 IoU 的改進，考慮了三個幾何因素：
    1. 重疊面積 (IoU)
    2. 中心點距離
    3. 長寬比一致性

    CIoU = IoU - (中心點距離² / 對角線距離²) - α * v
    其中 v 是長寬比一致性參數，α 是權重係數

    參數:
        box1: 形狀為 (N, 4) 的張量，格式為 [x1, y1, x2, y2]
        box2: 形狀為 (N, 4) 的張量，格式為 [x1, y1, x2, y2]
        eps: 防止除零的小數值

    返回:
        CIoU 值
    """
    # 計算交集
    interX1 = torch.max(box1[:, 0], box2[:, 0])
    interY1 = torch.max(box1[:, 1], box2[:, 1])
    interX2 = torch.min(box1[:, 2], box2[:, 2])
    interY2 = torch.min(box1[:, 3], box2[:, 3])
    interArea = (interX2 - interX1).clamp(min=0) * (interY2 - interY1).clamp(min=0)

    # 計算各自面積
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    unionArea = area1 + area2 - interArea + eps

    # 計算 IoU
    iou = interArea / unionArea

    # 計算最小包圍框的對角線距離
    convexX1 = torch.min(box1[:, 0], box2[:, 0])
    convexY1 = torch.min(box1[:, 1], box2[:, 1])
    convexX2 = torch.max(box1[:, 2], box2[:, 2])
    convexY2 = torch.max(box1[:, 3], box2[:, 3])
    convexDiag = (convexX2 - convexX1) ** 2 + (convexY2 - convexY1) ** 2 + eps

    # 計算中心點距離
    center1X = (box1[:, 0] + box1[:, 2]) / 2
    center1Y = (box1[:, 1] + box1[:, 3]) / 2
    center2X = (box2[:, 0] + box2[:, 2]) / 2
    center2Y = (box2[:, 1] + box2[:, 3]) / 2
    centerDist = (center1X - center2X) ** 2 + (center1Y - center2Y) ** 2

    # 計算長寬比一致性 v
    w1, h1 = box1[:, 2] - box1[:, 0], box1[:, 3] - box1[:, 1]
    w2, h2 = box2[:, 2] - box2[:, 0], box2[:, 3] - box2[:, 1]
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)

    # 計算 α 權重
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    # CIoU = IoU - 中心點懲罰 - 長寬比懲罰
    ciou = iou - centerDist / convexDiag - alpha * v

    return ciou


def nonMaxSuppression(
    predictions: torch.Tensor,
    confThreshold: float = 0.25,
    iouThreshold: float = 0.45,
    classes: Optional[List[int]] = None,
    maxDetections: int = 300
) -> List[torch.Tensor]:
    """
    執行非極大值抑制 (Non-Maximum Suppression, NMS)

    NMS 是物件檢測中的重要後處理步驟。當模型對同一物體產生多個重疊的預測框時，
    我們使用 NMS 來保留最佳的預測並移除冗餘的框。

    NMS 的步驟：
    1. 按照信心分數排序所有預測框
    2. 選擇分數最高的框
    3. 移除與該框 IoU 超過閾值的所有其他框
    4. 重複步驟 2-3 直到沒有剩餘的框

    參數:
        predictions: 模型輸出，形狀為 (batch_size, num_predictions, 5 + num_classes)
                    每個預測包含: [x, y, w, h, objectness, class1_prob, class2_prob, ...]
        confThreshold: 信心閾值，低於此值的預測會被過濾
        iouThreshold: IoU 閾值，高於此值的重疊框會被抑制
        classes: 如果指定，只保留這些類別的檢測結果
        maxDetections: 每張圖片的最大檢測數量

    返回:
        每張圖片的檢測結果列表，每個元素的形狀為 (num_detections, 6)
        格式: [x1, y1, x2, y2, confidence, class_id]
    """
    # 預測格式: [x, y, w, h, obj_conf, cls1, cls2, ...]
    batchSize = predictions.shape[0]
    numClasses = predictions.shape[2] - 5

    # 候選框的信心閾值（考慮物件性和類別機率的組合）
    candidateMask = predictions[..., 4] > confThreshold

    output = [torch.zeros((0, 6), device=predictions.device)] * batchSize

    for imageIdx in range(batchSize):
        # 取得該圖片的有效預測
        pred = predictions[imageIdx]
        pred = pred[candidateMask[imageIdx]]

        if not pred.shape[0]:
            continue

        # 計算最終信心分數 = 物件性 × 類別機率
        pred[:, 5:] *= pred[:, 4:5]  # conf = obj_conf * cls_conf

        # 將 xywh 轉換為 xyxy 格式
        boxes = xywh2xyxyTorch(pred[:, :4])

        # 取得每個預測的最佳類別
        classConf, classId = pred[:, 5:].max(dim=1, keepdim=True)

        # 過濾低信心的預測
        validMask = classConf.squeeze() > confThreshold
        if classes is not None:
            validMask &= torch.isin(classId.squeeze(), torch.tensor(classes, device=pred.device))

        # 組合: [x1, y1, x2, y2, confidence, class_id]
        detections = torch.cat([boxes, classConf, classId.float()], dim=1)[validMask]

        if not detections.shape[0]:
            continue

        # 執行 NMS（按類別分別處理）
        # 使用 torchvision 的 batched_nms，它會根據類別分別處理
        keepIndices = torchvision.ops.batched_nms(
            detections[:, :4],  # 邊界框
            detections[:, 4],   # 分數
            detections[:, 5],   # 類別 ID
            iouThreshold
        )

        # 限制最大檢測數量
        if keepIndices.shape[0] > maxDetections:
            keepIndices = keepIndices[:maxDetections]

        output[imageIdx] = detections[keepIndices]

    return output


def clipBoxes(boxes: torch.Tensor, imageShape: Tuple[int, int]) -> torch.Tensor:
    """
    將邊界框裁剪到圖片邊界內

    確保邊界框的座標不會超出圖片範圍。這在進行資料增強或後處理時很重要。

    參數:
        boxes: 邊界框，形狀為 (N, 4)，格式為 [x1, y1, x2, y2]
        imageShape: 圖片形狀 (height, width)

    返回:
        裁剪後的邊界框
    """
    height, width = imageShape
    boxes[:, 0].clamp_(0, width)   # x1
    boxes[:, 1].clamp_(0, height)  # y1
    boxes[:, 2].clamp_(0, width)   # x2
    boxes[:, 3].clamp_(0, height)  # y2
    return boxes


def scaleBoxes(
    originalShape: Tuple[int, int],
    boxes: torch.Tensor,
    targetShape: Tuple[int, int]
) -> torch.Tensor:
    """
    將邊界框從一個圖片尺寸縮放到另一個尺寸

    這在推論時很有用，因為模型的輸入尺寸可能與原始圖片不同。
    我們需要將模型輸出的邊界框轉換回原始圖片的座標系統。

    參數:
        originalShape: 原始形狀 (height, width)，這是模型輸入的尺寸
        boxes: 邊界框，格式為 [x1, y1, x2, y2]
        targetShape: 目標形狀 (height, width)，這是要轉換到的尺寸

    返回:
        縮放後的邊界框
    """
    # 計算縮放比例
    gain = min(originalShape[0] / targetShape[0], originalShape[1] / targetShape[1])

    # 計算填充量（letterbox 填充）
    padX = (originalShape[1] - targetShape[1] * gain) / 2
    padY = (originalShape[0] - targetShape[0] * gain) / 2

    # 移除填充並縮放
    boxes[:, [0, 2]] -= padX
    boxes[:, [1, 3]] -= padY
    boxes[:, :4] /= gain

    # 裁剪到目標圖片邊界
    clipBoxes(boxes, targetShape)

    return boxes


def colorList() -> List[Tuple[int, int, int]]:
    """
    生成用於視覺化的顏色列表

    返回:
        80 個不同顏色的 RGB 值列表
    """
    colors = []
    for i in range(80):
        # 使用 HSV 色彩空間生成均勻分布的顏色
        hue = i / 80
        # 轉換到 RGB
        r = int(255 * abs(math.sin(hue * 2 * math.pi)))
        g = int(255 * abs(math.sin((hue + 0.33) * 2 * math.pi)))
        b = int(255 * abs(math.sin((hue + 0.67) * 2 * math.pi)))
        colors.append((r, g, b))
    return colors


def getLatestRun(searchDir: str = 'runs/train') -> str:
    """
    取得最新的訓練輸出目錄

    參數:
        searchDir: 搜尋目錄

    返回:
        最新的實驗目錄路徑
    """
    lastDirs = glob.glob(f'{searchDir}/*/')
    if not lastDirs:
        return ''
    return max(lastDirs, key=os.path.getmtime)


def cleanString(s: str) -> str:
    """
    清理字串中的特殊字元

    參數:
        s: 輸入字串

    返回:
        清理後的字串
    """
    return ''.join(c for c in s if c.isalnum() or c in '._-')


class TimeTracker:
    """
    時間追蹤器類別

    用於追蹤程式執行時間，幫助分析效能瓶頸。
    """

    def __init__(self):
        """初始化時間追蹤器"""
        self.startTime = time.time()
        self.checkpoints = {}

    def checkpoint(self, name: str) -> float:
        """
        記錄檢查點

        參數:
            name: 檢查點名稱

        返回:
            從開始到現在的時間（秒）
        """
        elapsed = time.time() - self.startTime
        self.checkpoints[name] = elapsed
        return elapsed

    def getElapsed(self) -> float:
        """
        取得總經過時間

        返回:
            經過的秒數
        """
        return time.time() - self.startTime

    def reset(self) -> None:
        """重置計時器"""
        self.startTime = time.time()
        self.checkpoints = {}
