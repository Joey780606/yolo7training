"""
資料集模組 (Dataset Module)

此模組處理 YOLO 訓練所需的資料載入和預處理，包括：
- 影像和標籤檔案的讀取
- 資料增強的應用
- PyTorch DataLoader 的配置

YOLO 標籤格式:
每張圖片對應一個 .txt 標籤檔案，每行格式為:
<class_id> <x_center> <y_center> <width> <height>

座標都是正規化的值 (0.0 ~ 1.0)，相對於圖片尺寸。
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .augmentations import (
    letterbox, augmentHSV, horizontalFlip, verticalFlip,
    randomPerspective, loadMosaic, mixUp
)


class YoloDataset(Dataset):
    """
    YOLO 格式資料集類別

    這個類別負責：
    1. 載入影像和標籤
    2. 應用資料增強
    3. 將資料轉換為模型需要的格式

    目錄結構範例:
    ```
    data/
    ├── images/
    │   ├── train/
    │   │   ├── img001.jpg
    │   │   └── img002.jpg
    │   └── val/
    │       └── img003.jpg
    └── labels/
        ├── train/
        │   ├── img001.txt
        │   └── img002.txt
        └── val/
            └── img003.txt
    ```

    屬性:
        imgFiles: 影像檔案路徑列表
        labelFiles: 標籤檔案路徑列表
        imageSize: 輸入影像大小
        augment: 是否啟用增強
    """

    def __init__(
        self,
        imagesPath: str,
        imageSize: int = 640,
        batchSize: int = 16,
        augment: bool = True,
        hyperParams: Optional[Dict] = None,
        stride: int = 32,
        pad: float = 0.0,
        prefix: str = ''
    ):
        """
        初始化資料集

        參數:
            imagesPath: 影像目錄路徑
            imageSize: 輸入影像大小
            batchSize: 批次大小
            augment: 是否啟用資料增強
            hyperParams: 超參數字典（包含增強設定）
            stride: 步長（確保尺寸是其倍數）
            pad: 填充比例
            prefix: 日誌前綴
        """
        self.imageSize = imageSize
        self.augment = augment
        self.stride = stride
        self.prefix = prefix

        # 預設超參數
        self.hyperParams = hyperParams or {
            'hsv_h': 0.015,      # HSV 色調增強
            'hsv_s': 0.7,       # HSV 飽和度增強
            'hsv_v': 0.4,       # HSV 明度增強
            'degrees': 0.0,     # 旋轉角度
            'translate': 0.1,   # 平移比例
            'scale': 0.5,       # 縮放範圍
            'shear': 0.0,       # 剪切角度
            'perspective': 0.0, # 透視變換
            'flipud': 0.0,      # 垂直翻轉機率
            'fliplr': 0.5,      # 水平翻轉機率
            'mosaic': 1.0,      # Mosaic 機率
            'mixup': 0.0        # MixUp 機率
        }

        # ============= 掃描影像檔案 =============
        self.imgFiles = []
        supportedFormats = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']

        imagesPath = Path(imagesPath)

        if imagesPath.is_file():
            # 如果是檔案，讀取每行作為路徑
            with open(imagesPath, 'r') as f:
                self.imgFiles = [x.strip() for x in f.readlines() if x.strip()]
        else:
            # 如果是目錄，掃描所有影像
            for ext in supportedFormats:
                self.imgFiles.extend(list(imagesPath.glob(f'*{ext}')))
                self.imgFiles.extend(list(imagesPath.glob(f'*{ext.upper()}')))

        self.imgFiles = sorted([str(f) for f in self.imgFiles])
        numImages = len(self.imgFiles)

        if numImages == 0:
            raise ValueError(f'在 {imagesPath} 找不到影像檔案')

        print(f'{prefix}掃描到 {numImages} 張影像')

        # ============= 取得對應的標籤檔案 =============
        # 將 images 路徑轉換為 labels 路徑
        self.labelFiles = self._img2Label(self.imgFiles)

        # ============= 快取標籤 =============
        self.labels = []
        self.shapes = []
        numMissing = 0
        numEmpty = 0

        print(f'{prefix}載入標籤中...')
        for imgFile, labelFile in tqdm(zip(self.imgFiles, self.labelFiles), total=numImages):
            # 取得影像尺寸
            try:
                img = cv2.imread(imgFile)
                if img is None:
                    raise ValueError(f'無法讀取影像: {imgFile}')
                self.shapes.append(img.shape[:2])  # [height, width]
            except Exception as e:
                print(f'警告: {e}')
                self.shapes.append((640, 640))

            # 讀取標籤
            try:
                if os.path.isfile(labelFile):
                    with open(labelFile, 'r') as f:
                        lines = f.readlines()

                    label = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            classId = int(parts[0])
                            x, y, w, h = map(float, parts[1:5])
                            label.append([classId, x, y, w, h])

                    if len(label) > 0:
                        self.labels.append(np.array(label, dtype=np.float32))
                    else:
                        self.labels.append(np.zeros((0, 5), dtype=np.float32))
                        numEmpty += 1
                else:
                    self.labels.append(np.zeros((0, 5), dtype=np.float32))
                    numMissing += 1

            except Exception as e:
                print(f'警告: 讀取標籤錯誤 {labelFile}: {e}')
                self.labels.append(np.zeros((0, 5), dtype=np.float32))

        self.shapes = np.array(self.shapes)

        print(f'{prefix}標籤統計:')
        print(f'  - 有效標籤: {numImages - numMissing - numEmpty}')
        print(f'  - 空白標籤: {numEmpty}')
        print(f'  - 缺失標籤: {numMissing}')

        # 計算批次索引（用於 Mosaic）
        self.batchIndices = np.floor(np.arange(numImages) / batchSize).astype(int)

    def _img2Label(self, imgPaths: List[str]) -> List[str]:
        """
        將影像路徑轉換為標籤路徑

        規則：將路徑中的 'images' 替換為 'labels'，副檔名改為 .txt

        參數:
            imgPaths: 影像路徑列表

        返回:
            標籤路徑列表
        """
        labelPaths = []
        for imgPath in imgPaths:
            # 替換目錄名
            labelPath = imgPath.replace('/images/', '/labels/')
            labelPath = labelPath.replace('\\images\\', '\\labels\\')

            # 替換副檔名
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']:
                labelPath = labelPath.replace(ext, '.txt')
                labelPath = labelPath.replace(ext.upper(), '.txt')

            labelPaths.append(labelPath)

        return labelPaths

    def __len__(self) -> int:
        """返回資料集大小"""
        return len(self.imgFiles)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str, Tuple]:
        """
        取得一個訓練樣本

        參數:
            index: 樣本索引

        返回:
            (影像張量, 標籤張量, 影像路徑, 原始尺寸)
        """
        hyperParams = self.hyperParams

        # ============= Mosaic 增強 =============
        mosaic = self.augment and random.random() < hyperParams.get('mosaic', 0.0)

        if mosaic:
            # 選擇 4 張圖片
            indices = [index] + random.choices(range(len(self)), k=3)
            images = []
            targetsList = []

            for idx in indices:
                img, targets, _ = self._loadImageAndLabel(idx)
                images.append(img)
                targetsList.append(targets)

            # 執行 Mosaic
            img, targets = loadMosaic(images, targetsList, self.imageSize)

            # 隨機透視變換
            img, targets = randomPerspective(
                img, targets,
                degrees=hyperParams.get('degrees', 0.0),
                translate=hyperParams.get('translate', 0.1),
                scale=hyperParams.get('scale', 0.5),
                shear=hyperParams.get('shear', 0.0),
                perspective=hyperParams.get('perspective', 0.0)
            )

        else:
            # 單圖載入
            img, targets, (h0, w0) = self._loadImageAndLabel(index)

            # Letterbox 縮放
            img, ratio, pad = letterbox(img, (self.imageSize, self.imageSize), auto=False)

            # 更新標籤座標
            if len(targets) > 0:
                # 座標仍然是正規化的，不需要調整
                pass

            # 隨機透視變換
            if self.augment:
                img, targets = randomPerspective(
                    img, targets,
                    degrees=hyperParams.get('degrees', 0.0),
                    translate=hyperParams.get('translate', 0.1),
                    scale=hyperParams.get('scale', 0.5),
                    shear=hyperParams.get('shear', 0.0),
                    perspective=hyperParams.get('perspective', 0.0)
                )

        # ============= 其他增強 =============
        if self.augment:
            # HSV 色彩增強
            img = augmentHSV(
                img,
                hGain=hyperParams.get('hsv_h', 0.015),
                sGain=hyperParams.get('hsv_s', 0.7),
                vGain=hyperParams.get('hsv_v', 0.4)
            )

            # 水平翻轉
            if random.random() < hyperParams.get('fliplr', 0.5):
                img, targets = horizontalFlip(img, targets)

            # 垂直翻轉
            if random.random() < hyperParams.get('flipud', 0.0):
                img, targets = verticalFlip(img, targets)

            # MixUp
            if random.random() < hyperParams.get('mixup', 0.0):
                idx2 = random.randint(0, len(self) - 1)
                img2, targets2, _ = self._loadImageAndLabel(idx2)
                img2, _, _ = letterbox(img2, (self.imageSize, self.imageSize), auto=False)
                alpha = np.random.beta(8.0, 8.0)
                img, targets = mixUp(img, targets, img2, targets2, alpha)

        # ============= 轉換為張量 =============
        # 影像: HWC -> CHW, BGR -> RGB, 正規化到 [0, 1]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255.0

        # 標籤
        numLabels = len(targets)
        if numLabels > 0:
            labelsOut = torch.zeros((numLabels, 6))
            labelsOut[:, 1:] = torch.from_numpy(targets)
        else:
            labelsOut = torch.zeros((0, 6))

        return img, labelsOut, self.imgFiles[index], self.shapes[index]

    def _loadImageAndLabel(self, index: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        載入單張影像和標籤

        參數:
            index: 樣本索引

        返回:
            (影像, 標籤, 原始尺寸)
        """
        # 載入影像
        imgPath = self.imgFiles[index]
        img = cv2.imread(imgPath)

        if img is None:
            raise ValueError(f'無法讀取影像: {imgPath}')

        h0, w0 = img.shape[:2]

        # 載入標籤
        targets = self.labels[index].copy()

        return img, targets, (h0, w0)

    @staticmethod
    def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, List, List]:
        """
        自訂批次組合函數

        將多個樣本組合成一個批次。
        標籤需要特殊處理，因為每張圖片的物體數量不同。

        參數:
            batch: 樣本列表

        返回:
            (堆疊的影像, 合併的標籤, 路徑列表, 尺寸列表)
        """
        images, labels, paths, shapes = zip(*batch)

        # 堆疊影像
        images = torch.stack(images, dim=0)

        # 合併標籤，添加批次索引
        for i, label in enumerate(labels):
            label[:, 0] = i  # 設定批次索引

        labels = torch.cat(labels, dim=0)

        return images, labels, list(paths), list(shapes)


def createDataLoader(
    path: str,
    imageSize: int = 640,
    batchSize: int = 16,
    stride: int = 32,
    hyperParams: Optional[Dict] = None,
    augment: bool = True,
    shuffle: bool = True,
    numWorkers: int = 8,
    pin_memory: bool = True,
    prefix: str = ''
) -> Tuple[DataLoader, YoloDataset]:
    """
    建立 DataLoader

    參數:
        path: 資料集路徑
        imageSize: 影像大小
        batchSize: 批次大小
        stride: 步長
        hyperParams: 超參數
        augment: 是否增強
        shuffle: 是否打亂
        numWorkers: 工作行程數
        pin_memory: 是否鎖定記憶體
        prefix: 日誌前綴

    返回:
        (DataLoader, Dataset)
    """
    dataset = YoloDataset(
        imagesPath=path,
        imageSize=imageSize,
        batchSize=batchSize,
        augment=augment,
        hyperParams=hyperParams,
        stride=stride,
        prefix=prefix
    )

    # 計算實際 workers 數量
    numWorkers = min(numWorkers, os.cpu_count() or 1)

    dataLoader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=shuffle,
        num_workers=numWorkers,
        pin_memory=pin_memory,
        collate_fn=YoloDataset.collate_fn
    )

    return dataLoader, dataset


def loadDataConfig(configPath: str) -> Dict:
    """
    載入資料集設定檔

    參數:
        configPath: 設定檔路徑

    返回:
        設定字典
    """
    import yaml

    try:
        with open(configPath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f'找不到資料設定檔: {configPath}')
    except yaml.YAMLError as e:
        raise ValueError(f'資料設定檔格式錯誤: {e}')
    except Exception as e:
        raise RuntimeError(f'載入資料設定檔時發生錯誤: {e}')

    # 解析路徑
    configDir = Path(configPath).parent

    # 處理訓練集路徑
    if 'train' in config:
        trainPath = Path(config['train'])
        if not trainPath.is_absolute():
            config['train'] = str(configDir / trainPath)

    # 處理驗證集路徑
    if 'val' in config:
        valPath = Path(config['val'])
        if not valPath.is_absolute():
            config['val'] = str(configDir / valPath)

    # 處理測試集路徑
    if 'test' in config:
        testPath = Path(config['test'])
        if not testPath.is_absolute():
            config['test'] = str(configDir / testPath)

    return config
