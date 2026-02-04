"""
YOLOv7 主模型模組 (Main YOLO Model Module)

此模組定義了完整的 YOLOv7 模型，整合了：
- Backbone (骨幹網路): 特徵提取
- Neck (頸部網路): 多尺度特徵融合
- Head (檢測頭): 物件檢測輸出

使用方式:
    model = YOLOv7(numClasses=80)
    predictions = model(images)  # images: [batch, 3, 640, 640]

訓練模式:
    返回原始預測，供損失函數使用

推論模式:
    返回處理後的預測結果，可直接進行 NMS
"""

import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import yaml
import torch
import torch.nn as nn

from .backbone import YOLOv7Backbone, YOLOv7TinyBackbone
from .neck import FPNPAN, TinyFPNPAN
from .head import YOLOHead, Detect


class YOLOv7(nn.Module):
    """
    YOLOv7 物件檢測模型

    YOLOv7 是一個高效的單階段物件檢測器，具有以下特點：
    1. E-ELAN 骨幹網路，高效的特徵提取
    2. FPN + PAN 頸部網路，多尺度特徵融合
    3. 基於錨點框的檢測頭，支援多尺度檢測

    模型架構:
    ```
    輸入影像 [B, 3, H, W]
            ↓
    ┌─────────────────┐
    │    Backbone     │  特徵提取
    │   (E-ELAN)      │  P3: H/8, P4: H/16, P5: H/32
    └─────────────────┘
            ↓
    ┌─────────────────┐
    │      Neck       │  特徵融合
    │   (FPN + PAN)   │  雙向特徵金字塔
    └─────────────────┘
            ↓
    ┌─────────────────┐
    │      Head       │  檢測輸出
    │    (Detect)     │  邊界框 + 物件性 + 類別
    └─────────────────┘
            ↓
    預測結果 [B, N, 5 + numClasses]
    ```

    屬性:
        numClasses: 類別數量
        backbone: 骨幹網路
        neck: 頸部網路
        head: 檢測頭
    """

    def __init__(
        self,
        numClasses: int = 80,
        anchors: Optional[List[List[float]]] = None,
        modelType: str = 'yolov7',
        widthMultiple: float = 1.0,
        depthMultiple: float = 1.0
    ):
        """
        初始化 YOLOv7 模型

        參數:
            numClasses: 類別數量（COCO 為 80）
            anchors: 自訂錨點框尺寸
            modelType: 模型類型 ('yolov7' 或 'yolov7tiny')
            widthMultiple: 寬度倍數，控制通道數
            depthMultiple: 深度倍數，控制網路深度
        """
        super().__init__()

        self.numClasses = numClasses
        self.modelType = modelType.lower()

        # ============= 建立骨幹網路 =============
        if self.modelType == 'yolov7tiny':
            self.backbone = YOLOv7TinyBackbone(widthMultiple=widthMultiple)
        else:
            self.backbone = YOLOv7Backbone(
                widthMultiple=widthMultiple,
                depthMultiple=depthMultiple
            )

        # 取得骨幹網路的輸出通道數
        backboneChannels = self.backbone.getOutputChannels()

        # ============= 建立頸部網路 =============
        if self.modelType == 'yolov7tiny':
            self.neck = TinyFPNPAN(
                inChannels=backboneChannels,
                widthMultiple=widthMultiple
            )
        else:
            self.neck = FPNPAN(
                inChannels=backboneChannels,
                widthMultiple=widthMultiple
            )

        # 取得頸部網路的輸出通道數
        neckChannels = self.neck.getOutputChannels()

        # ============= 建立檢測頭 =============
        self.head = YOLOHead(
            numClasses=numClasses,
            anchors=anchors,
            inChannels=[neckChannels['p3'], neckChannels['p4'], neckChannels['p5']]
        )

        # 初始化權重
        self._initializeWeights()

    def _initializeWeights(self) -> None:
        """
        初始化模型權重

        使用適當的初始化方法可以加速訓練收斂。
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming 初始化，適合 ReLU/SiLU 激活函數
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> Union[Tuple[torch.Tensor, List], List]:
        """
        前向傳播

        參數:
            x: 輸入影像張量，形狀 [batch, 3, height, width]
               高度和寬度應該是 32 的倍數（例如 640×640）

        返回:
            訓練模式: 原始預測列表，每個元素形狀為
                     [batch, numAnchors, H, W, 5 + numClasses]
            推論模式: (處理後的預測, 原始預測列表)
                     處理後的預測形狀: [batch, totalPredictions, 5 + numClasses]
        """
        # 骨幹網路: 提取多尺度特徵
        p3, p4, p5 = self.backbone(x)

        # 頸部網路: 特徵融合
        p3, p4, p5 = self.neck(p3, p4, p5)

        # 檢測頭: 生成預測
        return self.head([p3, p4, p5])

    def fuse(self) -> 'YOLOv7':
        """
        融合模型層以加速推論

        將 Conv + BatchNorm 融合成單一卷積層，
        將 RepConv 的多分支結構融合成單一卷積。

        返回:
            融合後的模型（self）
        """
        print('融合模型層...')

        for module in self.modules():
            # 融合 Conv + BN
            if hasattr(module, 'conv') and hasattr(module, 'bn'):
                # 這裡可以實現 Conv+BN 融合
                pass

            # 融合 RepConv
            if hasattr(module, 'fuseRepConv'):
                module.fuseRepConv()

        return self

    def info(self, verbose: bool = False) -> None:
        """
        顯示模型資訊

        參數:
            verbose: 是否顯示詳細資訊
        """
        numParams = sum(p.numel() for p in self.parameters())
        numGrads = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f'\n{"="*50}')
        print(f'YOLOv7 模型資訊')
        print(f'{"="*50}')
        print(f'模型類型: {self.modelType}')
        print(f'類別數量: {self.numClasses}')
        print(f'參數總數: {numParams:,}')
        print(f'可訓練參數: {numGrads:,}')
        print(f'模型大小: {numParams * 4 / (1024 ** 2):.2f} MB (FP32)')
        print(f'{"="*50}\n')

        if verbose:
            print(self)


def loadYoloConfig(configPath: str) -> Dict[str, Any]:
    """
    載入 YOLO 設定檔

    參數:
        configPath: 設定檔路徑（YAML 格式）

    返回:
        設定字典
    """
    try:
        with open(configPath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f'找不到設定檔: {configPath}')
    except yaml.YAMLError as e:
        raise ValueError(f'設定檔格式錯誤: {e}')
    except Exception as e:
        raise RuntimeError(f'載入設定檔時發生錯誤: {e}')


def buildModel(
    modelConfig: Union[str, Dict[str, Any]],
    dataConfig: Optional[Union[str, Dict[str, Any]]] = None
) -> YOLOv7:
    """
    根據設定檔建立 YOLOv7 模型

    參數:
        modelConfig: 模型設定檔路徑或設定字典
        dataConfig: 資料集設定檔路徑或設定字典（用於取得類別數量）

    返回:
        建立的 YOLOv7 模型
    """
    # 載入模型設定
    if isinstance(modelConfig, str):
        modelCfg = loadYoloConfig(modelConfig)
    else:
        modelCfg = modelConfig

    # 載入資料設定（如果提供）
    if dataConfig is not None:
        if isinstance(dataConfig, str):
            dataCfg = loadYoloConfig(dataConfig)
        else:
            dataCfg = dataConfig
        numClasses = dataCfg.get('nc', 80)
    else:
        numClasses = modelCfg.get('nc', 80)

    # 取得模型參數
    anchors = modelCfg.get('anchors', None)
    widthMultiple = modelCfg.get('widthMultiple', 1.0)
    depthMultiple = modelCfg.get('depthMultiple', 1.0)

    # 判斷模型類型
    if 'tiny' in str(modelConfig).lower():
        modelType = 'yolov7tiny'
    else:
        modelType = 'yolov7'

    # 建立模型
    model = YOLOv7(
        numClasses=numClasses,
        anchors=anchors,
        modelType=modelType,
        widthMultiple=widthMultiple,
        depthMultiple=depthMultiple
    )

    return model


def loadWeights(
    model: YOLOv7,
    weightsPath: str,
    device: torch.device = torch.device('cpu'),
    strict: bool = True
) -> YOLOv7:
    """
    載入預訓練權重

    參數:
        model: YOLOv7 模型
        weightsPath: 權重檔案路徑
        device: 載入到的裝置
        strict: 是否嚴格匹配權重名稱

    返回:
        載入權重後的模型
    """
    try:
        checkpoint = torch.load(weightsPath, map_location=device)
    except FileNotFoundError:
        raise FileNotFoundError(f'找不到權重檔案: {weightsPath}')
    except Exception as e:
        raise RuntimeError(f'載入權重時發生錯誤: {e}')

    # 支援不同格式的檢查點
    if 'model' in checkpoint:
        stateDict = checkpoint['model']
        if hasattr(stateDict, 'state_dict'):
            stateDict = stateDict.state_dict()
    elif 'state_dict' in checkpoint:
        stateDict = checkpoint['state_dict']
    else:
        stateDict = checkpoint

    # 載入權重
    try:
        model.load_state_dict(stateDict, strict=strict)
        print(f'成功載入權重: {weightsPath}')
    except RuntimeError as e:
        if not strict:
            # 非嚴格模式下，只載入匹配的權重
            modelDict = model.state_dict()
            matchedKeys = {k: v for k, v in stateDict.items() if k in modelDict and v.shape == modelDict[k].shape}
            modelDict.update(matchedKeys)
            model.load_state_dict(modelDict)
            print(f'部分載入權重: {len(matchedKeys)}/{len(modelDict)} 個參數')
        else:
            raise e

    return model


def countFlops(model: YOLOv7, inputSize: Tuple[int, int] = (640, 640)) -> float:
    """
    估算模型的 FLOPs (浮點運算次數)

    參數:
        model: YOLOv7 模型
        inputSize: 輸入尺寸 (height, width)

    返回:
        FLOPs 數量 (單位: G, 十億次運算)
    """
    try:
        from thop import profile
        dummyInput = torch.randn(1, 3, inputSize[0], inputSize[1])
        flops, params = profile(model, inputs=(dummyInput,), verbose=False)
        return flops / 1e9  # 轉換為 GFLOPs
    except ImportError:
        # 如果沒有安裝 thop，使用簡單估算
        numParams = sum(p.numel() for p in model.parameters())
        # 粗略估算: FLOPs ≈ 2 × params × H × W / 32
        flops = 2 * numParams * inputSize[0] * inputSize[1] / 32
        return flops / 1e9
    except Exception:
        return 0.0


# 模組匯出
__all__ = [
    'YOLOv7',
    'buildModel',
    'loadWeights',
    'loadYoloConfig',
    'countFlops'
]
