"""
檢測頭模組 (Detection Head Module)

檢測頭 (Detection Head) 是 YOLO 網路的最後部分，負責將特徵圖轉換為最終的檢測結果。
每個檢測頭在一個特定尺度上進行預測。

YOLO 的檢測原理:
1. 將輸入影像劃分為 S×S 的網格
2. 每個網格單元負責預測落在其中的物體
3. 每個網格使用多個錨點框 (Anchor Boxes) 來預測不同形狀的物體
4. 每個預測包含: 邊界框座標 (4) + 物件性分數 (1) + 類別機率 (num_classes)

預測格式:
- tx, ty: 相對於網格單元的中心偏移
- tw, th: 相對於錨點框的對數縮放
- objectness: 此位置存在物體的機率
- class_probs: 各類別的機率分布
"""

import math
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn

from .common import Conv, RepConv


class Detect(nn.Module):
    """
    YOLO 檢測頭

    這個模組整合了多個尺度的檢測結果。每個尺度有獨立的預測卷積層，
    但共享錨點框的設計和後處理邏輯。

    三個尺度對應不同大小的物體:
    - P3 (stride=8): 80×80 網格，檢測小物體
    - P4 (stride=16): 40×40 網格，檢測中物體
    - P5 (stride=32): 20×20 網格，檢測大物體

    屬性:
        numClasses: 類別數量
        numAnchors: 每個尺度的錨點框數量
        numOutputs: 每個錨點框的輸出數量 (4 + 1 + numClasses)
        anchors: 錨點框尺寸
        strides: 各尺度的步長
        detectLayers: 各尺度的檢測卷積層
    """

    def __init__(
        self,
        numClasses: int = 80,
        anchors: List[List[float]] = None,
        inChannels: List[int] = [128, 256, 512]
    ):
        """
        初始化檢測頭

        參數:
            numClasses: 類別數量
            anchors: 錨點框尺寸，每個尺度 3 個錨點框
                    格式: [[w1,h1, w2,h2, w3,h3], [...], [...]]
            inChannels: 各尺度輸入的通道數
        """
        super().__init__()

        # 預設錨點框（來自 COCO 資料集的聚類結果）
        if anchors is None:
            anchors = [
                [12, 16, 19, 36, 40, 28],         # P3 小物體
                [36, 75, 76, 55, 72, 146],        # P4 中物體
                [142, 110, 192, 243, 459, 401]   # P5 大物體
            ]

        self.numClasses = numClasses
        self.numDetectLayers = len(anchors)  # 檢測層數量（通常為 3）
        self.numAnchorsPerScale = len(anchors[0]) // 2  # 每個尺度的錨點數

        # 每個錨點框的輸出維度: [tx, ty, tw, th, obj, cls1, cls2, ...]
        self.numOutputsPerAnchor = 5 + numClasses

        # 各尺度的下採樣倍數
        self.strides = torch.tensor([8, 16, 32])

        # 註冊錨點框（會在 forward 時根據 stride 調整）
        self.register_buffer(
            'anchorsOriginal',
            torch.tensor(anchors).float().view(self.numDetectLayers, -1, 2)
        )

        # 建立各尺度的檢測卷積層
        # 輸出通道數 = 錨點數 × (5 + 類別數)
        numOutputChannels = self.numAnchorsPerScale * self.numOutputsPerAnchor
        self.detectLayers = nn.ModuleList([
            nn.Conv2d(inCh, numOutputChannels, 1) for inCh in inChannels
        ])

        # 用於推論時的網格快取
        self.gridCache: Dict[Tuple[int, int, int], torch.Tensor] = {}
        self.anchorGridCache: Dict[Tuple[int, int, int], torch.Tensor] = {}

        # 初始化偏置
        self._initializeBiases()

    def _initializeBiases(self) -> None:
        """
        初始化檢測層的偏置

        良好的偏置初始化可以加速訓練收斂。
        - 物件性偏置: 初始化為使初始預測約為 0.01
        - 類別偏置: 基於類別先驗機率初始化
        """
        for layer, stride in zip(self.detectLayers, self.strides):
            bias = layer.bias.view(self.numAnchorsPerScale, -1)

            # 物件性偏置: 讓初始預測接近 0.01
            # sigmoid(b) ≈ 0.01 -> b ≈ -4.6
            bias.data[:, 4] += math.log(8 / (640 / stride.item()) ** 2)

            # 類別偏置: 假設均勻分布
            bias.data[:, 5:] += math.log(0.6 / (self.numClasses - 0.99))

            layer.bias = nn.Parameter(bias.view(-1))

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        前向傳播

        參數:
            features: 各尺度的特徵圖列表 [p3, p4, p5]

        返回:
            如果是訓練模式:
                返回原始預測列表，供損失函數使用
            如果是推論模式:
                返回 (合併的預測結果, 原始預測列表)
                合併的預測形狀: [batch, num_all_predictions, 5 + numClasses]
        """
        outputs = []
        inferenceOutputs = []

        for i, (feature, layer) in enumerate(zip(features, self.detectLayers)):
            # 通過檢測卷積層
            # 輸入: [batch, inChannels, H, W]
            # 輸出: [batch, numAnchors * (5 + numClasses), H, W]
            prediction = layer(feature)

            batchSize, _, gridH, gridW = prediction.shape

            # 重塑預測張量
            # [batch, numAnchors * numOutputs, H, W] ->
            # [batch, numAnchors, numOutputs, H, W] ->
            # [batch, numAnchors, H, W, numOutputs]
            prediction = prediction.view(
                batchSize, self.numAnchorsPerScale, self.numOutputsPerAnchor, gridH, gridW
            ).permute(0, 1, 3, 4, 2).contiguous()

            outputs.append(prediction)

            if not self.training:
                # 推論模式: 將預測轉換為絕對座標
                inferenceOutput = self._makeInferencePrediction(prediction, i, gridH, gridW)
                inferenceOutputs.append(inferenceOutput)

        if self.training:
            return outputs
        else:
            # 合併所有尺度的預測
            # 每個 inferenceOutput: [batch, numAnchors * H * W, 5 + numClasses]
            return torch.cat(inferenceOutputs, dim=1), outputs

    def _makeInferencePrediction(
        self,
        prediction: torch.Tensor,
        layerIdx: int,
        gridH: int,
        gridW: int
    ) -> torch.Tensor:
        """
        將訓練格式的預測轉換為推論格式

        訓練格式: 相對座標（相對於網格和錨點）
        推論格式: 絕對座標（相對於原始影像）

        轉換公式:
        - bx = sigmoid(tx) * 2 - 0.5 + cx
        - by = sigmoid(ty) * 2 - 0.5 + cy
        - bw = (sigmoid(tw) * 2)^2 * anchor_w
        - bh = (sigmoid(th) * 2)^2 * anchor_h

        參數:
            prediction: 原始預測 [batch, numAnchors, H, W, numOutputs]
            layerIdx: 當前檢測層索引
            gridH: 網格高度
            gridW: 網格寬度

        返回:
            轉換後的預測 [batch, numAnchors * H * W, numOutputs]
        """
        device = prediction.device
        batchSize = prediction.shape[0]
        stride = self.strides[layerIdx].to(device)

        # 取得或建立網格座標快取
        cacheKey = (layerIdx, gridH, gridW)

        if cacheKey not in self.gridCache:
            # 建立網格座標
            # gridY: [H, W]，每個位置的 y 座標 (0, 1, 2, ..., H-1)
            # gridX: [H, W]，每個位置的 x 座標 (0, 1, 2, ..., W-1)
            gridY, gridX = torch.meshgrid(
                torch.arange(gridH, device=device),
                torch.arange(gridW, device=device),
                indexing='ij'
            )
            # 堆疊成 [1, 1, H, W, 2]
            grid = torch.stack([gridX, gridY], dim=-1).view(1, 1, gridH, gridW, 2).float()
            self.gridCache[cacheKey] = grid

            # 取得該尺度的錨點框，並根據 stride 縮放
            anchorGrid = self.anchorsOriginal[layerIdx].clone() / stride
            # 擴展成 [1, numAnchors, 1, 1, 2]
            anchorGrid = anchorGrid.view(1, self.numAnchorsPerScale, 1, 1, 2)
            self.anchorGridCache[cacheKey] = anchorGrid.to(device)

        grid = self.gridCache[cacheKey]
        anchorGrid = self.anchorGridCache[cacheKey]

        # 複製預測以避免修改原始張量
        output = prediction.clone()

        # 應用激活函數並轉換座標
        # xy: 使用改進的座標預測公式
        output[..., :2] = (output[..., :2].sigmoid() * 2 - 0.5 + grid) * stride

        # wh: 使用改進的尺寸預測公式
        output[..., 2:4] = (output[..., 2:4].sigmoid() * 2) ** 2 * anchorGrid * stride

        # objectness 和 class probabilities: sigmoid
        output[..., 4:] = output[..., 4:].sigmoid()

        # 重塑為 [batch, numAnchors * H * W, numOutputs]
        return output.view(batchSize, -1, self.numOutputsPerAnchor)


class ImplicitA(nn.Module):
    """
    隱式知識模組 A (Implicit Knowledge A)

    這是 YOLOv7 中的隱式知識學習模組，用於學習與輸入無關的特徵偏移。
    它通過可學習的參數來調整特徵，有助於提升檢測性能。

    屬性:
        implicit: 可學習的隱式張量
    """

    def __init__(self, channels: int, mean: float = 0.0, std: float = 0.02):
        """
        初始化隱式知識模組

        參數:
            channels: 通道數
            mean: 初始化平均值
            std: 初始化標準差
        """
        super().__init__()
        self.channels = channels
        self.implicit = nn.Parameter(torch.zeros(1, channels, 1, 1))
        nn.init.normal_(self.implicit, mean=mean, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播: 將隱式知識加到輸入特徵上

        參數:
            x: 輸入特徵

        返回:
            調整後的特徵
        """
        return x + self.implicit


class ImplicitM(nn.Module):
    """
    隱式知識模組 M (Implicit Knowledge M)

    與 ImplicitA 類似，但使用乘法而非加法來調整特徵。

    屬性:
        implicit: 可學習的隱式張量
    """

    def __init__(self, channels: int, mean: float = 1.0, std: float = 0.02):
        """
        初始化隱式知識模組

        參數:
            channels: 通道數
            mean: 初始化平均值（乘法預設為 1）
            std: 初始化標準差
        """
        super().__init__()
        self.channels = channels
        self.implicit = nn.Parameter(torch.ones(1, channels, 1, 1))
        nn.init.normal_(self.implicit, mean=mean, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播: 將隱式知識乘到輸入特徵上

        參數:
            x: 輸入特徵

        返回:
            調整後的特徵
        """
        return x * self.implicit


class YOLOHead(nn.Module):
    """
    完整的 YOLO 檢測頭（包含特徵處理和檢測）

    這個模組包含：
    1. RepConv 特徵處理層（每個尺度一個）
    2. Detect 檢測層

    屬性:
        repConvs: RepConv 層列表
        detect: 檢測模組
    """

    def __init__(
        self,
        numClasses: int = 80,
        anchors: Optional[List[List[float]]] = None,
        inChannels: List[int] = [128, 256, 512],
        useRepConv: bool = True
    ):
        """
        初始化 YOLO 檢測頭

        參數:
            numClasses: 類別數量
            anchors: 錨點框尺寸
            inChannels: 各尺度輸入通道數
            useRepConv: 是否使用 RepConv（推論時可融合）
        """
        super().__init__()

        # 特徵處理層
        if useRepConv:
            self.repConvs = nn.ModuleList([
                RepConv(inCh, inCh, 3, 1) for inCh in inChannels
            ])
        else:
            self.repConvs = nn.ModuleList([
                Conv(inCh, inCh, 3, 1) for inCh in inChannels
            ])

        # 檢測層
        self.detect = Detect(
            numClasses=numClasses,
            anchors=anchors,
            inChannels=inChannels
        )

    def forward(self, features: List[torch.Tensor]):
        """
        前向傳播

        參數:
            features: 各尺度特徵列表

        返回:
            檢測結果
        """
        # 通過 RepConv 處理
        processedFeatures = [repConv(f) for repConv, f in zip(self.repConvs, features)]

        # 通過檢測層
        return self.detect(processedFeatures)

    def fuseRepConv(self) -> None:
        """融合 RepConv 層以加速推論"""
        for layer in self.repConvs:
            if hasattr(layer, 'fuseRepConv'):
                layer.fuseRepConv()


def buildHead(config: Dict[str, Any], neckChannels: Dict[str, int]) -> nn.Module:
    """
    根據設定檔建立檢測頭

    參數:
        config: 模型設定字典
        neckChannels: 頸部網路的輸出通道數

    返回:
        檢測頭模組
    """
    numClasses = config.get('nc', 80)
    anchors = config.get('anchors', None)

    inChannels = [
        neckChannels['p3'],
        neckChannels['p4'],
        neckChannels['p5']
    ]

    return YOLOHead(
        numClasses=numClasses,
        anchors=anchors,
        inChannels=inChannels
    )
