"""
通用模組 (Common Modules)

此模組定義了構建 YOLOv7 所需的基本神經網路層和模組，包括：
- Conv: 標準卷積模組 (卷積 + 批次正規化 + 激活函數)
- Bottleneck: 瓶頸結構
- SPPCSPC: 空間金字塔池化跨階段部分連接
- E-ELAN: 擴展高效層聚合網路
- RepConv: 重參數化卷積

這些是深度學習物件檢測網路中最常用的構建區塊。
理解這些基本模組對於理解整個 YOLO 架構至關重要。
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(kernelSize: Union[int, Tuple[int, int]], padding: Optional[int] = None) -> int:
    """
    自動計算填充量以保持輸出尺寸不變 (same padding)

    當使用 stride=1 時，要讓輸出尺寸等於輸入尺寸，
    需要設定 padding = kernel_size // 2。

    參數:
        kernelSize: 卷積核大小
        padding: 指定的填充量（如果為 None 則自動計算）

    返回:
        計算出的填充量
    """
    if padding is None:
        # 如果是 tuple，取第一個值；否則直接使用
        if isinstance(kernelSize, int):
            padding = kernelSize // 2
        else:
            padding = kernelSize[0] // 2
    return padding


class Conv(nn.Module):
    """
    標準卷積模組 (Standard Convolution Module)

    這是 YOLO 中最基本的構建區塊，組合了三個操作：
    1. 2D 卷積 (Convolution): 提取空間特徵
    2. 批次正規化 (Batch Normalization): 加速訓練並穩定學習
    3. 激活函數 (Activation): 引入非線性

    為什麼要把這三個操作組合在一起？
    - 減少程式碼重複
    - 確保一致性
    - 方便模型修改

    屬性:
        conv: 卷積層
        bn: 批次正規化層
        act: 激活函數
    """

    def __init__(
        self,
        inChannels: int,
        outChannels: int,
        kernelSize: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        activation: bool = True
    ):
        """
        初始化卷積模組

        參數:
            inChannels: 輸入通道數
            outChannels: 輸出通道數
            kernelSize: 卷積核大小，預設 1
            stride: 步長，預設 1
            padding: 填充量，預設自動計算
            groups: 分組卷積的組數，預設 1（普通卷積）
            activation: 是否使用激活函數，預設 True
        """
        super().__init__()

        # 2D 卷積層
        # bias=False 因為後面有 BatchNorm，它會學習偏置
        self.conv = nn.Conv2d(
            inChannels,
            outChannels,
            kernelSize,
            stride,
            autopad(kernelSize, padding),
            groups=groups,
            bias=False
        )

        # 批次正規化層
        self.bn = nn.BatchNorm2d(outChannels)

        # 激活函數: SiLU (Sigmoid Linear Unit)
        # SiLU(x) = x * sigmoid(x)
        # SiLU 比 ReLU 更平滑，在 YOLO 中表現更好
        self.act = nn.SiLU() if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        參數:
            x: 輸入張量，形狀 [batch, channels, height, width]

        返回:
            輸出張量
        """
        return self.act(self.bn(self.conv(x)))

    def forwardFuse(self, x: torch.Tensor) -> torch.Tensor:
        """
        融合後的前向傳播（跳過 BatchNorm）

        在推論時，可以將 Conv 和 BN 融合，此時使用這個方法。

        參數:
            x: 輸入張量

        返回:
            輸出張量
        """
        return self.act(self.conv(x))


class DWConv(Conv):
    """
    深度可分離卷積 (Depthwise Separable Convolution)

    深度可分離卷積將標準卷積分解為兩步：
    1. 深度卷積 (Depthwise): 每個輸入通道單獨卷積
    2. 逐點卷積 (Pointwise): 1x1 卷積混合通道資訊

    這大幅減少了參數量和計算量：
    - 標準卷積: O(K² × Cin × Cout)
    - 深度可分離: O(K² × Cin + Cin × Cout)

    參數量減少比例: 約 K² 倍（K 為卷積核大小）
    """

    def __init__(
        self,
        inChannels: int,
        outChannels: int,
        kernelSize: int = 1,
        stride: int = 1,
        activation: bool = True
    ):
        """
        初始化深度可分離卷積

        參數:
            inChannels: 輸入通道數
            outChannels: 輸出通道數
            kernelSize: 卷積核大小
            stride: 步長
            activation: 是否使用激活函數
        """
        super().__init__(
            inChannels,
            outChannels,
            kernelSize,
            stride,
            groups=math.gcd(inChannels, outChannels),  # 使用最大公因數作為組數
            activation=activation
        )


class Bottleneck(nn.Module):
    """
    瓶頸結構 (Bottleneck Structure)

    瓶頸是深度學習中常用的設計模式，來自 ResNet。
    它先降低通道數（壓縮），再恢復（擴展），形成「瓶頸」形狀。

    結構: [通道數]
    輸入 [C] -> 1x1 Conv [C/2] -> 3x3 Conv [C/2] -> 1x1 Conv [C] -> 輸出

    優點：
    1. 減少參數量（在低維度做運算）
    2. 增加網路深度
    3. 殘差連接幫助梯度流動

    屬性:
        cv1: 第一個卷積（降維）
        cv2: 第二個卷積（特徵提取）
        addResidual: 是否使用殘差連接
    """

    def __init__(
        self,
        inChannels: int,
        outChannels: int,
        shortcut: bool = True,
        groups: int = 1,
        expansion: float = 0.5
    ):
        """
        初始化瓶頸結構

        參數:
            inChannels: 輸入通道數
            outChannels: 輸出通道數
            shortcut: 是否使用殘差連接（捷徑）
            groups: 分組卷積的組數
            expansion: 擴展比例，決定中間層的通道數
        """
        super().__init__()

        # 計算中間層（瓶頸處）的通道數
        hiddenChannels = int(outChannels * expansion)

        # 1x1 卷積降維
        self.cv1 = Conv(inChannels, hiddenChannels, 1, 1)

        # 3x3 卷積提取特徵
        self.cv2 = Conv(hiddenChannels, outChannels, 3, 1, groups=groups)

        # 只有當輸入輸出通道相同時才能使用殘差連接
        self.addResidual = shortcut and inChannels == outChannels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        如果啟用殘差連接: output = input + conv(input)
        否則: output = conv(input)

        參數:
            x: 輸入張量

        返回:
            輸出張量
        """
        if self.addResidual:
            return x + self.cv2(self.cv1(x))
        return self.cv2(self.cv1(x))


class SPP(nn.Module):
    """
    空間金字塔池化 (Spatial Pyramid Pooling)

    SPP 可以處理任意尺寸的輸入，並生成固定長度的輸出。
    它使用不同大小的池化核來捕獲不同尺度的特徵。

    工作原理:
    1. 輸入特徵經過多個不同大小的 MaxPool
    2. 所有池化結果與原始特徵拼接
    3. 這樣可以在不損失空間資訊的情況下擴大感受野

    例如使用 [5, 9, 13] 三個池化核:
    - 5x5 池化捕獲局部特徵
    - 9x9 池化捕獲中等範圍特徵
    - 13x13 池化捕獲較大範圍特徵

    屬性:
        cv1: 輸入卷積
        cv2: 輸出卷積
        maxPools: 不同大小的最大池化層列表
    """

    def __init__(self, inChannels: int, outChannels: int, kernelSizes: Tuple[int, ...] = (5, 9, 13)):
        """
        初始化 SPP 模組

        參數:
            inChannels: 輸入通道數
            outChannels: 輸出通道數
            kernelSizes: 池化核大小列表
        """
        super().__init__()

        hiddenChannels = inChannels // 2

        # 輸入卷積，降低通道數
        self.cv1 = Conv(inChannels, hiddenChannels, 1, 1)

        # 輸出卷積
        # 輸入通道數 = hiddenChannels * (1 + len(kernelSizes))
        # 因為原始特徵 + 各尺度池化結果
        self.cv2 = Conv(hiddenChannels * (len(kernelSizes) + 1), outChannels, 1, 1)

        # 建立不同大小的最大池化層
        # padding = k // 2 確保輸出尺寸不變
        self.maxPools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
            for k in kernelSizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        參數:
            x: 輸入張量

        返回:
            融合了多尺度特徵的輸出張量
        """
        x = self.cv1(x)

        # 拼接原始特徵和所有池化結果
        pooledFeatures = [x] + [mp(x) for mp in self.maxPools]

        return self.cv2(torch.cat(pooledFeatures, dim=1))


class SPPCSPC(nn.Module):
    """
    SPPCSPC (SPP + Cross Stage Partial Connection)

    這是 YOLOv7 中使用的空間金字塔池化模組，結合了：
    1. SPP: 多尺度特徵提取
    2. CSP: 跨階段部分連接，提高梯度流動效率

    CSP 結構的核心思想是將輸入分成兩部分：
    - 一部分直接傳遞（短路徑）
    - 一部分經過複雜運算（長路徑）
    最後將兩部分合併

    這樣可以減少計算量同時保持表達能力。

    屬性:
        cv1-cv7: 各個卷積層
        maxPools: 最大池化層
    """

    def __init__(
        self,
        inChannels: int,
        outChannels: int,
        expansion: float = 0.5,
        kernelSizes: Tuple[int, ...] = (5, 9, 13)
    ):
        """
        初始化 SPPCSPC 模組

        參數:
            inChannels: 輸入通道數
            outChannels: 輸出通道數
            expansion: 擴展比例
            kernelSizes: SPP 池化核大小
        """
        super().__init__()

        hiddenChannels = int(2 * outChannels * expansion)

        # CSP 第一分支（短路徑）
        self.cv1 = Conv(inChannels, hiddenChannels, 1, 1)

        # CSP 第二分支（長路徑）
        self.cv2 = Conv(inChannels, hiddenChannels, 1, 1)

        # SPP 前的卷積
        self.cv3 = Conv(hiddenChannels, hiddenChannels, 3, 1)
        self.cv4 = Conv(hiddenChannels, hiddenChannels, 1, 1)

        # SPP 池化層
        self.maxPools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
            for k in kernelSizes
        ])

        # SPP 後的卷積
        # 輸入 = hiddenChannels * (1 + 3) = hiddenChannels * 4
        self.cv5 = Conv(hiddenChannels * (len(kernelSizes) + 1), hiddenChannels, 1, 1)
        self.cv6 = Conv(hiddenChannels, hiddenChannels, 3, 1)

        # 輸出卷積，合併兩個分支
        # 輸入 = hiddenChannels * 2（來自兩個分支）
        self.cv7 = Conv(hiddenChannels * 2, outChannels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        參數:
            x: 輸入張量

        返回:
            輸出張量
        """
        # 短路徑
        shortPath = self.cv1(x)

        # 長路徑
        longPath = self.cv4(self.cv3(self.cv2(x)))

        # SPP 多尺度池化
        pooledFeatures = [longPath] + [mp(longPath) for mp in self.maxPools]
        longPath = self.cv6(self.cv5(torch.cat(pooledFeatures, dim=1)))

        # 合併兩個路徑
        return self.cv7(torch.cat([shortPath, longPath], dim=1))


class Concat(nn.Module):
    """
    特徵拼接模組 (Concatenation Module)

    沿著指定維度拼接多個張量。
    在 FPN/PAN 結構中用於融合不同層的特徵。

    屬性:
        dim: 拼接的維度
    """

    def __init__(self, dimension: int = 1):
        """
        初始化拼接模組

        參數:
            dimension: 拼接維度，預設為 1（通道維度）
        """
        super().__init__()
        self.dim = dimension

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        前向傳播

        參數:
            x: 張量列表

        返回:
            拼接後的張量
        """
        return torch.cat(x, dim=self.dim)


class Upsample(nn.Module):
    """
    上採樣模組 (Upsampling Module)

    將特徵圖放大到指定倍數。
    在 FPN 的自頂向下路徑中使用，將高層特徵上採樣後與低層特徵融合。

    使用最近鄰插值 (nearest neighbor interpolation)，
    這是最簡單且計算效率最高的上採樣方法。

    屬性:
        scaleFactor: 縮放倍數
    """

    def __init__(self, scaleFactor: int = 2, mode: str = 'nearest'):
        """
        初始化上採樣模組

        參數:
            scaleFactor: 放大倍數
            mode: 插值方法，'nearest' 或 'bilinear'
        """
        super().__init__()
        self.scaleFactor = scaleFactor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        參數:
            x: 輸入張量 [batch, channels, height, width]

        返回:
            上採樣後的張量 [batch, channels, height*scale, width*scale]
        """
        return F.interpolate(x, scale_factor=self.scaleFactor, mode=self.mode)


class MPConv(nn.Module):
    """
    MaxPool + Conv 下採樣模組 (MaxPool Convolution Module)

    這是 YOLOv7 中的下採樣策略，結合了：
    1. MaxPool 路徑: 保留最顯著的特徵
    2. Conv (stride=2) 路徑: 學習下採樣

    相比單純使用 stride=2 的卷積，這種方式：
    - 保留更多細節資訊
    - 增加特徵多樣性
    - 改善梯度流動

    屬性:
        mp: 最大池化層
        cv1: 第一個卷積
        cv2: 第二個卷積（stride=2）
        cv3: 合併後的卷積
    """

    def __init__(self, inChannels: int):
        """
        初始化 MPConv 模組

        參數:
            inChannels: 輸入通道數
        """
        super().__init__()

        hiddenChannels = inChannels // 2

        # MaxPool 路徑
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cv1 = Conv(inChannels, hiddenChannels, 1, 1)

        # Conv stride=2 路徑
        self.cv2 = Conv(inChannels, hiddenChannels, 1, 1)
        self.cv3 = Conv(hiddenChannels, hiddenChannels, 3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        參數:
            x: 輸入張量

        返回:
            下採樣後的張量（空間尺寸減半，通道數不變）
        """
        # MaxPool 路徑
        path1 = self.cv1(self.mp(x))

        # Conv stride=2 路徑
        path2 = self.cv3(self.cv2(x))

        # 拼接兩個路徑
        return torch.cat([path1, path2], dim=1)


class ELANBlock(nn.Module):
    """
    E-ELAN 區塊 (Extended Efficient Layer Aggregation Network Block)

    E-ELAN 是 YOLOv7 的核心創新之一。它通過以下方式提升效率：
    1. 使用多個分支並行處理
    2. 在最後聚合所有分支的特徵
    3. 控制最短和最長梯度路徑

    結構示意:
    輸入 ──┬── cv1 ─────────────────────┐
           │                            │
           ├── cv2 ─┬── cv3 ─┬── cv4 ──┤
           │        │        │          │
           │        │        └─ cv5 ────┤
           │        │                   │
           │        └─────────────────┐ │
           │                          │ │
           └─────────────────────────[Concat]── cv6 ─> 輸出

    這種設計確保了梯度可以通過多條路徑流動，有助於訓練更深的網路。

    屬性:
        cv1-cv6: 各個卷積層
    """

    def __init__(self, inChannels: int, outChannels: int, hiddenChannels: int):
        """
        初始化 E-ELAN 區塊

        參數:
            inChannels: 輸入通道數
            outChannels: 輸出通道數
            hiddenChannels: 中間層通道數
        """
        super().__init__()

        # 第一分支：直接 1x1 卷積
        self.cv1 = Conv(inChannels, hiddenChannels, 1, 1)

        # 第二分支：1x1 卷積後接多個 3x3 卷積
        self.cv2 = Conv(inChannels, hiddenChannels, 1, 1)
        self.cv3 = Conv(hiddenChannels, hiddenChannels, 3, 1)
        self.cv4 = Conv(hiddenChannels, hiddenChannels, 3, 1)
        self.cv5 = Conv(hiddenChannels, hiddenChannels, 3, 1)

        # 輸出卷積：聚合所有分支
        # 輸入通道數 = hiddenChannels * 5（來自 cv1, cv2, cv3, cv4, cv5）
        self.cv6 = Conv(hiddenChannels * 5, outChannels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        參數:
            x: 輸入張量

        返回:
            輸出張量
        """
        # 計算各分支
        branch1 = self.cv1(x)
        branch2 = self.cv2(x)
        branch3 = self.cv3(branch2)
        branch4 = self.cv4(branch3)
        branch5 = self.cv5(branch4)

        # 聚合所有分支
        return self.cv6(torch.cat([branch1, branch2, branch3, branch4, branch5], dim=1))


class RepConv(nn.Module):
    """
    重參數化卷積 (Re-parameterizable Convolution)

    RepConv 是一種訓練-推論解耦的設計：
    - 訓練時: 使用多分支結構（3x3 + 1x1 + identity）
    - 推論時: 融合成單一 3x3 卷積

    訓練時的多分支結構可以：
    1. 增加模型容量
    2. 引入隱式正則化
    3. 改善梯度流動

    推論時融合後：
    1. 保持完全相同的數學等價性
    2. 減少記憶體存取
    3. 提升推論速度

    屬性:
        conv1: 3x3 卷積
        conv2: 1x1 卷積
        bn: 恆等映射的批次正規化（如果適用）
        act: 激活函數
    """

    def __init__(
        self,
        inChannels: int,
        outChannels: int,
        kernelSize: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        deploy: bool = False
    ):
        """
        初始化 RepConv

        參數:
            inChannels: 輸入通道數
            outChannels: 輸出通道數
            kernelSize: 主卷積核大小
            stride: 步長
            padding: 填充量
            groups: 分組卷積組數
            deploy: 是否為部署模式（融合後的單一卷積）
        """
        super().__init__()

        self.deploy = deploy
        self.groups = groups
        self.inChannels = inChannels
        self.outChannels = outChannels

        padding = autopad(kernelSize, padding)

        if deploy:
            # 部署模式：單一卷積
            self.repConv = nn.Conv2d(
                inChannels, outChannels, kernelSize, stride, padding, groups=groups, bias=True
            )
        else:
            # 訓練模式：多分支結構
            # 恆等映射分支（只在輸入輸出通道相同且 stride=1 時使用）
            self.bn = nn.BatchNorm2d(inChannels) if inChannels == outChannels and stride == 1 else None

            # 1x1 卷積分支
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(inChannels, outChannels, 1, stride, 0, groups=groups, bias=False),
                nn.BatchNorm2d(outChannels)
            )

            # 3x3 卷積分支（主分支）
            self.conv3x3 = nn.Sequential(
                nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding, groups=groups, bias=False),
                nn.BatchNorm2d(outChannels)
            )

        # 激活函數
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        參數:
            x: 輸入張量

        返回:
            輸出張量
        """
        if self.deploy:
            return self.act(self.repConv(x))

        # 訓練模式：合併三個分支
        out = self.conv3x3(x) + self.conv1x1(x)
        if self.bn is not None:
            out = out + self.bn(x)

        return self.act(out)

    def fuseRepConv(self) -> None:
        """
        融合重參數化卷積

        將訓練時的多分支結構融合成單一卷積，用於推論部署。
        這個操作是數學等價的，不會影響模型輸出。
        """
        if self.deploy:
            return

        # 取得 3x3 卷積的權重和偏置
        kernel3x3, bias3x3 = self._getEquivalentKernelBias(
            self.conv3x3[0], self.conv3x3[1]
        )

        # 取得 1x1 卷積的權重和偏置（填充成 3x3）
        kernel1x1, bias1x1 = self._getEquivalentKernelBias(
            self.conv1x1[0], self.conv1x1[1]
        )
        kernel1x1 = F.pad(kernel1x1, [1, 1, 1, 1])

        # 取得恆等映射的等效權重
        if self.bn is not None:
            kernelIdentity, biasIdentity = self._getEquivalentKernelBias(
                None, self.bn
            )
        else:
            kernelIdentity = torch.zeros_like(kernel3x3)
            biasIdentity = torch.zeros_like(bias3x3)

        # 合併所有分支
        fusedKernel = kernel3x3 + kernel1x1 + kernelIdentity
        fusedBias = bias3x3 + bias1x1 + biasIdentity

        # 建立融合後的卷積層
        self.repConv = nn.Conv2d(
            self.inChannels, self.outChannels, 3, self.conv3x3[0].stride[0],
            1, groups=self.groups, bias=True
        )
        self.repConv.weight.data = fusedKernel
        self.repConv.bias.data = fusedBias

        # 刪除訓練時的分支
        self.__delattr__('conv3x3')
        self.__delattr__('conv1x1')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')

        self.deploy = True

    def _getEquivalentKernelBias(self, conv, bn):
        """
        計算卷積+BN 的等效權重和偏置

        參數:
            conv: 卷積層（可為 None，表示恆等映射）
            bn: 批次正規化層

        返回:
            (等效權重, 等效偏置)
        """
        if conv is None:
            # 恆等映射：建立單位卷積核
            kernelSize = 3
            kernel = torch.zeros(
                self.inChannels, self.inChannels // self.groups, kernelSize, kernelSize,
                device=bn.weight.device
            )
            for i in range(self.inChannels):
                kernel[i, i % (self.inChannels // self.groups), 1, 1] = 1
        else:
            kernel = conv.weight

        # 融合 BN 參數
        runningMean = bn.running_mean
        runningVar = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = (runningVar + eps).sqrt()
        fusedKernel = kernel * (gamma / std).reshape(-1, 1, 1, 1)
        fusedBias = beta - runningMean * gamma / std

        return fusedKernel, fusedBias
