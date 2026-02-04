"""
資料增強模組 (Data Augmentation Module)

資料增強是提升物件檢測模型泛化能力的關鍵技術。
通過對訓練影像進行各種變換，可以人工增加訓練資料的多樣性，
使模型更能適應真實世界中的各種情況。

此模組實現了以下增強方法：
- Mosaic: 將 4 張圖片拼接成一張
- MixUp: 混合兩張圖片
- HSV 變換: 調整色調、飽和度、明度
- 翻轉: 水平/垂直翻轉
- 縮放和填充: Letterbox 處理
- 隨機仿射變換: 旋轉、縮放、平移、剪切
"""

import math
import random
from typing import Tuple, List, Optional, Union

import cv2
import numpy as np


def letterbox(
    image: np.ndarray,
    targetSize: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scaleFill: bool = False,
    scaleUp: bool = True,
    stride: int = 32
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    調整影像大小並填充至目標尺寸 (Letterbox Resize)

    Letterbox 是 YOLO 中常用的圖片預處理方法：
    1. 按比例縮放影像，使最長邊等於目標尺寸
    2. 用灰色填充短邊，使影像達到目標尺寸
    3. 保持原始長寬比，避免影像變形

    為什麼要用 Letterbox？
    - 神經網路需要固定大小的輸入
    - 直接縮放會改變物體比例，影響檢測效果
    - 填充保持了原始影像的長寬比

    參數:
        image: 輸入影像 (BGR 格式)
        targetSize: 目標尺寸 (height, width)
        color: 填充顏色 (BGR)
        auto: 是否自動調整填充以符合 stride
        scaleFill: 是否直接縮放填滿（不保持長寬比）
        scaleUp: 是否允許放大小圖片
        stride: 步長（確保輸出尺寸是 stride 的倍數）

    返回:
        (處理後的影像, 縮放比例, 填充量)
    """
    # 取得原始尺寸
    originalShape = image.shape[:2]  # [height, width]
    targetH, targetW = targetSize

    # 計算縮放比例
    ratio = min(targetH / originalShape[0], targetW / originalShape[1])

    # 如果不允許放大，限制比例
    if not scaleUp:
        ratio = min(ratio, 1.0)

    # 計算縮放後的尺寸
    newUnpadH = int(round(originalShape[0] * ratio))
    newUnpadW = int(round(originalShape[1] * ratio))

    # 計算需要的填充量
    padH = targetH - newUnpadH
    padW = targetW - newUnpadW

    if auto:
        # 自動模式：最小填充，確保是 stride 的倍數
        padH = padH % stride
        padW = padW % stride
    elif scaleFill:
        # 直接縮放填滿
        padH, padW = 0, 0
        newUnpadH, newUnpadW = targetH, targetW
        ratio = targetW / originalShape[1], targetH / originalShape[0]

    # 將填充平均分配到兩側
    padTop = padH // 2
    padBottom = padH - padTop
    padLeft = padW // 2
    padRight = padW - padLeft

    # 縮放影像
    if originalShape[::-1] != (newUnpadW, newUnpadH):
        image = cv2.resize(image, (newUnpadW, newUnpadH), interpolation=cv2.INTER_LINEAR)

    # 添加填充
    image = cv2.copyMakeBorder(
        image, padTop, padBottom, padLeft, padRight,
        cv2.BORDER_CONSTANT, value=color
    )

    return image, (ratio, ratio), (padLeft, padTop)


def randomPerspective(
    image: np.ndarray,
    targets: np.ndarray = None,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
    borderColor: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    隨機透視/仿射變換

    這個函數執行複合的幾何變換，包括：
    - 旋轉 (Rotation)
    - 縮放 (Scale)
    - 平移 (Translation)
    - 剪切 (Shear)
    - 透視 (Perspective)

    這些變換模擬了真實世界中相機角度和距離的變化。

    參數:
        image: 輸入影像
        targets: 標籤 [class, x, y, w, h]（正規化座標）
        degrees: 旋轉角度範圍
        translate: 平移比例範圍
        scale: 縮放比例範圍
        shear: 剪切角度範圍
        perspective: 透視變換強度
        borderColor: 邊界填充顏色

    返回:
        (變換後的影像, 變換後的標籤)
    """
    height, width = image.shape[:2]

    # 中心點
    centerX = width / 2
    centerY = height / 2

    # ============= 建立變換矩陣 =============

    # 透視變換矩陣 (如果啟用)
    perspectiveMatrix = np.eye(3)
    if perspective:
        perspectiveMatrix[2, 0] = random.uniform(-perspective, perspective)
        perspectiveMatrix[2, 1] = random.uniform(-perspective, perspective)

    # 旋轉和縮放矩陣
    rotationAngle = random.uniform(-degrees, degrees)
    scaleFactor = random.uniform(1 - scale, 1 + scale)

    rotationMatrix = cv2.getRotationMatrix2D((centerX, centerY), rotationAngle, scaleFactor)
    rotationMatrixFull = np.eye(3)
    rotationMatrixFull[:2] = rotationMatrix

    # 剪切矩陣
    shearMatrix = np.eye(3)
    shearMatrix[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    shearMatrix[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    # 平移矩陣
    translateMatrix = np.eye(3)
    translateMatrix[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    translateMatrix[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    # 組合變換矩陣: T @ S @ R @ P
    transformMatrix = translateMatrix @ shearMatrix @ rotationMatrixFull @ perspectiveMatrix

    # 應用變換
    if perspective:
        image = cv2.warpPerspective(
            image, transformMatrix, (width, height),
            borderValue=borderColor
        )
    else:
        image = cv2.warpAffine(
            image, transformMatrix[:2], (width, height),
            borderValue=borderColor
        )

    # ============= 變換標籤座標 =============
    if targets is not None and len(targets) > 0:
        numTargets = len(targets)

        # 將 xywh (正規化) 轉換為 xyxy (絕對座標)
        # targets: [class, x_center, y_center, width, height]
        xywhNorm = targets[:, 1:5].copy()

        # 轉換為角點座標
        # [x1, y1, x2, y2, x3, y3, x4, y4] 四個角點
        corners = np.zeros((numTargets, 8))
        corners[:, 0] = (xywhNorm[:, 0] - xywhNorm[:, 2] / 2) * width  # x1
        corners[:, 1] = (xywhNorm[:, 1] - xywhNorm[:, 3] / 2) * height  # y1
        corners[:, 2] = (xywhNorm[:, 0] + xywhNorm[:, 2] / 2) * width  # x2
        corners[:, 3] = (xywhNorm[:, 1] - xywhNorm[:, 3] / 2) * height  # y2
        corners[:, 4] = (xywhNorm[:, 0] + xywhNorm[:, 2] / 2) * width  # x3
        corners[:, 5] = (xywhNorm[:, 1] + xywhNorm[:, 3] / 2) * height  # y3
        corners[:, 6] = (xywhNorm[:, 0] - xywhNorm[:, 2] / 2) * width  # x4
        corners[:, 7] = (xywhNorm[:, 1] + xywhNorm[:, 3] / 2) * height  # y4

        # 將角點轉換為齊次座標並應用變換
        corners = corners.reshape(-1, 2)
        cornersHom = np.hstack([corners, np.ones((corners.shape[0], 1))])
        cornersTransformed = (transformMatrix @ cornersHom.T).T
        cornersTransformed = cornersTransformed[:, :2] / cornersTransformed[:, 2:3]
        corners = cornersTransformed.reshape(-1, 8)

        # 計算新的邊界框
        x = corners[:, [0, 2, 4, 6]]
        y = corners[:, [1, 3, 5, 7]]
        newBoxes = np.zeros((numTargets, 4))
        newBoxes[:, 0] = x.min(axis=1)  # x1
        newBoxes[:, 1] = y.min(axis=1)  # y1
        newBoxes[:, 2] = x.max(axis=1)  # x2
        newBoxes[:, 3] = y.max(axis=1)  # y2

        # 裁剪到影像邊界
        newBoxes[:, [0, 2]] = newBoxes[:, [0, 2]].clip(0, width)
        newBoxes[:, [1, 3]] = newBoxes[:, [1, 3]].clip(0, height)

        # 過濾無效的框（面積過小或超出邊界）
        validMask = (newBoxes[:, 2] - newBoxes[:, 0] > 2) & (newBoxes[:, 3] - newBoxes[:, 1] > 2)

        # 轉換回 xywh 正規化格式
        targets = targets[validMask]
        newBoxes = newBoxes[validMask]

        if len(newBoxes) > 0:
            targets[:, 1] = ((newBoxes[:, 0] + newBoxes[:, 2]) / 2) / width  # x_center
            targets[:, 2] = ((newBoxes[:, 1] + newBoxes[:, 3]) / 2) / height  # y_center
            targets[:, 3] = (newBoxes[:, 2] - newBoxes[:, 0]) / width  # width
            targets[:, 4] = (newBoxes[:, 3] - newBoxes[:, 1]) / height  # height

    return image, targets if targets is not None else np.zeros((0, 5))


def augmentHSV(
    image: np.ndarray,
    hGain: float = 0.5,
    sGain: float = 0.5,
    vGain: float = 0.5
) -> np.ndarray:
    """
    HSV 色彩空間增強

    HSV (Hue, Saturation, Value) 增強是一種常用的色彩增強技術：
    - Hue (色調): 調整顏色（紅、綠、藍等）
    - Saturation (飽和度): 調整顏色的鮮豔程度
    - Value (明度): 調整亮度

    這種增強可以模擬不同的光照條件和相機設定。

    參數:
        image: 輸入影像 (BGR 格式)
        hGain: 色調增益範圍 [0, 1]
        sGain: 飽和度增益範圍 [0, 1]
        vGain: 明度增益範圍 [0, 1]

    返回:
        增強後的影像
    """
    if hGain == 0 and sGain == 0 and vGain == 0:
        return image

    # 生成隨機增益
    r = np.random.uniform(-1, 1, 3) * [hGain, sGain, vGain] + 1

    # 轉換到 HSV 色彩空間
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    dtype = image.dtype

    # 建立查詢表 (LUT) 以加速處理
    lutHue = ((np.arange(0, 256) * r[0]) % 180).astype(dtype)
    lutSat = np.clip(np.arange(0, 256) * r[1], 0, 255).astype(dtype)
    lutVal = np.clip(np.arange(0, 256) * r[2], 0, 255).astype(dtype)

    # 應用 LUT
    hue = cv2.LUT(hue, lutHue)
    sat = cv2.LUT(sat, lutSat)
    val = cv2.LUT(val, lutVal)

    # 合併並轉換回 BGR
    imageHSV = cv2.merge((hue, sat, val))
    return cv2.cvtColor(imageHSV, cv2.COLOR_HSV2BGR)


def horizontalFlip(
    image: np.ndarray,
    targets: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    水平翻轉

    將影像左右翻轉，同時調整標籤座標。

    參數:
        image: 輸入影像
        targets: 標籤 [class, x, y, w, h]

    返回:
        (翻轉後的影像, 翻轉後的標籤)
    """
    image = np.fliplr(image)

    if targets is not None and len(targets) > 0:
        # x_center = 1 - x_center
        targets[:, 1] = 1 - targets[:, 1]

    return image, targets if targets is not None else np.zeros((0, 5))


def verticalFlip(
    image: np.ndarray,
    targets: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    垂直翻轉

    將影像上下翻轉，同時調整標籤座標。

    參數:
        image: 輸入影像
        targets: 標籤 [class, x, y, w, h]

    返回:
        (翻轉後的影像, 翻轉後的標籤)
    """
    image = np.flipud(image)

    if targets is not None and len(targets) > 0:
        # y_center = 1 - y_center
        targets[:, 2] = 1 - targets[:, 2]

    return image, targets if targets is not None else np.zeros((0, 5))


def loadMosaic(
    images: List[np.ndarray],
    targetsList: List[np.ndarray],
    imageSize: int = 640
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mosaic 資料增強

    Mosaic 是 YOLOv4 引入的強大增強技術：
    將 4 張圖片拼接成一張大圖，然後隨機裁剪回原始大小。

    優點：
    1. 豐富背景多樣性
    2. 增加小物體的數量
    3. 批次正規化時看到更多樣本
    4. 讓模型學習在不同上下文中識別物體

    參數:
        images: 4 張輸入影像列表
        targetsList: 4 組標籤列表，每個是 [class, x, y, w, h]
        imageSize: 輸出影像大小

    返回:
        (拼接後的影像, 合併後的標籤)
    """
    assert len(images) == 4, "Mosaic 需要恰好 4 張圖片"

    mosaicSize = imageSize * 2
    mosaicImage = np.full((mosaicSize, mosaicSize, 3), 114, dtype=np.uint8)

    # 隨機選擇中心點（確定 4 張圖的分界線）
    centerX = int(random.uniform(imageSize * 0.5, imageSize * 1.5))
    centerY = int(random.uniform(imageSize * 0.5, imageSize * 1.5))

    allTargets = []

    # 四個位置：左上、右上、左下、右下
    positions = [
        (0, centerY, 0, centerX),               # 左上: y範圍 [0, centerY], x範圍 [0, centerX]
        (0, centerY, centerX, mosaicSize),      # 右上
        (centerY, mosaicSize, 0, centerX),      # 左下
        (centerY, mosaicSize, centerX, mosaicSize)  # 右下
    ]

    for i, (img, targets) in enumerate(zip(images, targetsList)):
        y1, y2, x1, x2 = positions[i]
        regionH = y2 - y1
        regionW = x2 - x1

        # 調整圖片大小以填滿區域
        h, w = img.shape[:2]
        scale = min(regionH / h, regionW / w)
        newH, newW = int(h * scale), int(w * scale)

        # 計算偏移量（讓圖片在區域內居中或隨機放置）
        offsetY = random.randint(0, max(0, regionH - newH))
        offsetX = random.randint(0, max(0, regionW - newW))

        # 縮放並放置圖片
        imgResized = cv2.resize(img, (newW, newH))
        mosaicImage[y1 + offsetY:y1 + offsetY + newH, x1 + offsetX:x1 + offsetX + newW] = imgResized

        # 調整標籤座標
        if len(targets) > 0:
            targetsAdjusted = targets.copy()

            # 轉換座標到 mosaic 空間
            # 原始座標是 [0, 1] 範圍，先轉換為絕對座標
            targetsAdjusted[:, 1] = targets[:, 1] * newW + x1 + offsetX  # x_center
            targetsAdjusted[:, 2] = targets[:, 2] * newH + y1 + offsetY  # y_center
            targetsAdjusted[:, 3] = targets[:, 3] * newW  # width
            targetsAdjusted[:, 4] = targets[:, 4] * newH  # height

            allTargets.append(targetsAdjusted)

    # 合併所有標籤
    if allTargets:
        allTargets = np.vstack(allTargets)
    else:
        allTargets = np.zeros((0, 5))

    # 隨機裁剪 mosaic 到 imageSize
    cropX = random.randint(0, mosaicSize - imageSize)
    cropY = random.randint(0, mosaicSize - imageSize)

    mosaicImage = mosaicImage[cropY:cropY + imageSize, cropX:cropX + imageSize]

    # 調整標籤到裁剪後的座標
    if len(allTargets) > 0:
        # 減去裁剪偏移
        allTargets[:, 1] -= cropX
        allTargets[:, 2] -= cropY

        # 裁剪邊界框到影像範圍內
        x1 = allTargets[:, 1] - allTargets[:, 3] / 2
        y1 = allTargets[:, 2] - allTargets[:, 4] / 2
        x2 = allTargets[:, 1] + allTargets[:, 3] / 2
        y2 = allTargets[:, 2] + allTargets[:, 4] / 2

        x1 = np.clip(x1, 0, imageSize)
        y1 = np.clip(y1, 0, imageSize)
        x2 = np.clip(x2, 0, imageSize)
        y2 = np.clip(y2, 0, imageSize)

        # 更新邊界框並過濾無效的
        newW = x2 - x1
        newH = y2 - y1
        validMask = (newW > 2) & (newH > 2)

        allTargets = allTargets[validMask]
        x1 = x1[validMask]
        y1 = y1[validMask]
        newW = newW[validMask]
        newH = newH[validMask]

        if len(allTargets) > 0:
            # 轉換回正規化的 xywh 格式
            allTargets[:, 1] = (x1 + newW / 2) / imageSize
            allTargets[:, 2] = (y1 + newH / 2) / imageSize
            allTargets[:, 3] = newW / imageSize
            allTargets[:, 4] = newH / imageSize

    return mosaicImage, allTargets


def mixUp(
    image1: np.ndarray,
    targets1: np.ndarray,
    image2: np.ndarray,
    targets2: np.ndarray,
    alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MixUp 資料增強

    MixUp 將兩張圖片按比例混合，並合併它們的標籤。
    這是一種正則化技術，可以提高模型的泛化能力。

    混合公式: mixed = α * image1 + (1 - α) * image2

    參數:
        image1: 第一張影像
        targets1: 第一張影像的標籤
        image2: 第二張影像
        targets2: 第二張影像的標籤
        alpha: 混合比例（從 Beta 分布取樣）

    返回:
        (混合後的影像, 合併後的標籤)
    """
    # 確保兩張圖片尺寸相同
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # 混合影像
    mixedImage = (alpha * image1 + (1 - alpha) * image2).astype(np.uint8)

    # 合併標籤
    if len(targets1) > 0 and len(targets2) > 0:
        mixedTargets = np.vstack([targets1, targets2])
    elif len(targets1) > 0:
        mixedTargets = targets1
    elif len(targets2) > 0:
        mixedTargets = targets2
    else:
        mixedTargets = np.zeros((0, 5))

    return mixedImage, mixedTargets


def cutout(
    image: np.ndarray,
    targets: np.ndarray = None,
    scales: List[float] = [0.5, 0.25, 0.125],
    ratio: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cutout 資料增強

    Cutout 在圖片上隨機遮擋矩形區域，
    迫使模型學習使用其他特徵來識別物體。

    參數:
        image: 輸入影像
        targets: 標籤
        scales: 遮擋區域相對於影像的大小比例列表
        ratio: 應用 cutout 的機率

    返回:
        (處理後的影像, 標籤)
    """
    if random.random() > ratio:
        return image, targets if targets is not None else np.zeros((0, 5))

    h, w = image.shape[:2]

    for scale in scales:
        # 計算遮擋區域大小
        cutH = int(h * scale)
        cutW = int(w * scale)

        # 隨機選擇位置
        x = random.randint(0, w - cutW)
        y = random.randint(0, h - cutH)

        # 應用遮擋（使用灰色）
        image[y:y + cutH, x:x + cutW] = 114

    return image, targets if targets is not None else np.zeros((0, 5))


class Albumentations:
    """
    Albumentations 增強封裝類別

    如果安裝了 albumentations 庫，可以使用更多高級增強方法。
    這個類別提供了一個統一的介面。
    """

    def __init__(self):
        """初始化增強器"""
        self.transform = None
        try:
            import albumentations as A
            self.transform = A.Compose([
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        except ImportError:
            pass

    def __call__(
        self,
        image: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        應用增強

        參數:
            image: 輸入影像 (BGR)
            targets: 標籤 [class, x, y, w, h]

        返回:
            (增強後的影像, 標籤)
        """
        if self.transform is None or len(targets) == 0:
            return image, targets

        # 轉換為 RGB (albumentations 使用 RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 準備 albumentations 格式的標籤
        bboxes = targets[:, 1:].tolist()
        classLabels = targets[:, 0].tolist()

        try:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=classLabels
            )
            image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
            if transformed['bboxes']:
                bboxes = np.array(transformed['bboxes'])
                classLabels = np.array(transformed['class_labels']).reshape(-1, 1)
                targets = np.hstack([classLabels, bboxes])
        except Exception:
            pass

        return image, targets
