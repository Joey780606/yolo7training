"""
工具模組 (Utilities Module)

此模組包含 YOLOv7 訓練和推論所需的各種工具函式：
- general: 通用工具函式
- torchUtils: PyTorch 輔助函式
- datasets: 資料集處理
- augmentations: 資料增強
- loss: 損失函數
- metrics: 評估指標

使用方式:
    from utils import createDataLoader, ComputeLoss, MetricsCalculator
"""

from .general import (
    setRandomSeed,
    checkFileExists,
    checkDirExists,
    makeDirectory,
    incrementPath,
    xyxy2xywh,
    xywh2xyxy,
    boxIou,
    boxCiou,
    nonMaxSuppression,
    clipBoxes,
    scaleBoxes,
    colorList,
)

from .torchUtils import (
    selectDevice,
    countParameters,
    modelInfo,
    initializeWeights,
    ModelEMA,
    getCosineScheduler,
    saveCheckpoint,
    loadCheckpoint,
)

from .datasets import (
    YoloDataset,
    createDataLoader,
    loadDataConfig,
)

from .augmentations import (
    letterbox,
    augmentHSV,
    horizontalFlip,
    verticalFlip,
    randomPerspective,
    loadMosaic,
    mixUp,
)

from .loss import ComputeLoss, FocalLoss

from .metrics import (
    MetricsCalculator,
    ConfusionMatrix,
    computeAp,
    apPerClass,
    fitnessScore,
)

__all__ = [
    # general
    'setRandomSeed',
    'checkFileExists',
    'checkDirExists',
    'makeDirectory',
    'incrementPath',
    'xyxy2xywh',
    'xywh2xyxy',
    'boxIou',
    'boxCiou',
    'nonMaxSuppression',
    'clipBoxes',
    'scaleBoxes',
    'colorList',
    # torchUtils
    'selectDevice',
    'countParameters',
    'modelInfo',
    'initializeWeights',
    'ModelEMA',
    'getCosineScheduler',
    'saveCheckpoint',
    'loadCheckpoint',
    # datasets
    'YoloDataset',
    'createDataLoader',
    'loadDataConfig',
    # augmentations
    'letterbox',
    'augmentHSV',
    'horizontalFlip',
    'verticalFlip',
    'randomPerspective',
    'loadMosaic',
    'mixUp',
    # loss
    'ComputeLoss',
    'FocalLoss',
    # metrics
    'MetricsCalculator',
    'ConfusionMatrix',
    'computeAp',
    'apPerClass',
    'fitnessScore',
]
