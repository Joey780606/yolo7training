"""
骨幹網路模組 (Backbone Network Module)

骨幹網路 (Backbone) 是物件檢測網路的基礎，負責從原始影像中提取特徵。
YOLOv7 使用 E-ELAN (Extended Efficient Layer Aggregation Network) 作為骨幹網路。

主要功能：
1. 逐步降低空間解析度（下採樣）
2. 逐步增加通道數（提升特徵表達能力）
3. 在不同尺度上輸出特徵圖（P3, P4, P5）

這些多尺度特徵會被傳遞給 Neck（頸部網路）進行進一步的特徵融合。
"""

from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn

from .common import Conv, ELANBlock, MPConv, SPPCSPC


class YOLOv7Backbone(nn.Module):
    """
    YOLOv7 骨幹網路

    網路結構：
    - Stem: 初始特徵提取
    - Stage 1-2: 早期特徵處理
    - Stage 3 (P3/8): 小物體特徵（8 倍下採樣）
    - Stage 4 (P4/16): 中物體特徵（16 倍下採樣）
    - Stage 5 (P5/32): 大物體特徵（32 倍下採樣）
    - SPPCSPC: 空間金字塔池化，擴大感受野

    屬性:
        stem: 輸入處理層
        stage1-5: 各階段特徵提取模組
        sppcspc: 空間金字塔池化模組
    """

    def __init__(
        self,
        inChannels: int = 3,
        baseChannels: int = 64,
        depthMultiple: float = 1.0,
        widthMultiple: float = 1.0
    ):
        """
        初始化骨幹網路

        參數:
            inChannels: 輸入通道數（RGB 影像為 3）
            baseChannels: 基礎通道數
            depthMultiple: 深度倍數，控制網路深度
            widthMultiple: 寬度倍數，控制通道數
        """
        super().__init__()

        # 根據寬度倍數調整通道數
        def ch(channels: int) -> int:
            """計算調整後的通道數"""
            return max(int(channels * widthMultiple), 1)

        # ============= Stem =============
        # 輸入: [B, 3, H, W] -> 輸出: [B, 64, H/2, W/2]
        self.stem = nn.Sequential(
            Conv(inChannels, ch(32), 3, 1),  # 3x3 卷積，保持尺寸
            Conv(ch(32), ch(64), 3, 2),       # 3x3 卷積，下採樣 2x
            Conv(ch(64), ch(64), 3, 1)        # 3x3 卷積，特徵處理
        )

        # ============= Stage 2 (P2/4) =============
        # 輸入: [B, 64, H/2, W/2] -> 輸出: [B, 256, H/4, W/4]
        self.stage2 = nn.Sequential(
            Conv(ch(64), ch(128), 3, 2),               # 下採樣
            ELANBlock(ch(128), ch(256), ch(64))        # E-ELAN 特徵提取
        )

        # ============= Stage 3 (P3/8) - 小物體特徵 =============
        # 輸入: [B, 256, H/4, W/4] -> 輸出: [B, 512, H/8, W/8]
        self.stage3 = nn.Sequential(
            MPConv(ch(256)),                           # MaxPool+Conv 下採樣
            ELANBlock(ch(256), ch(512), ch(128))       # E-ELAN 特徵提取
        )
        self.p3Channels = ch(512)

        # ============= Stage 4 (P4/16) - 中物體特徵 =============
        # 輸入: [B, 512, H/8, W/8] -> 輸出: [B, 1024, H/16, W/16]
        self.stage4 = nn.Sequential(
            MPConv(ch(512)),                           # MaxPool+Conv 下採樣
            ELANBlock(ch(512), ch(1024), ch(256))      # E-ELAN 特徵提取
        )
        self.p4Channels = ch(1024)

        # ============= Stage 5 (P5/32) - 大物體特徵 =============
        # 輸入: [B, 1024, H/16, W/16] -> 輸出: [B, 1024, H/32, W/32]
        self.stage5 = nn.Sequential(
            MPConv(ch(1024)),                          # MaxPool+Conv 下採樣
            ELANBlock(ch(1024), ch(1024), ch(256))     # E-ELAN 特徵提取
        )

        # ============= SPPCSPC =============
        # 輸入: [B, 1024, H/32, W/32] -> 輸出: [B, 512, H/32, W/32]
        self.sppcspc = SPPCSPC(ch(1024), ch(512))
        self.p5Channels = ch(512)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向傳播

        參數:
            x: 輸入影像張量，形狀 [batch, 3, height, width]

        返回:
            三個尺度的特徵圖 (p3, p4, p5)：
            - p3: [batch, 512, H/8, W/8] - 小物體特徵
            - p4: [batch, 1024, H/16, W/16] - 中物體特徵
            - p5: [batch, 512, H/32, W/32] - 大物體特徵
        """
        # 初始特徵提取
        x = self.stem(x)

        # Stage 2
        x = self.stage2(x)

        # Stage 3 - 輸出 P3 特徵
        x = self.stage3(x)
        p3 = x

        # Stage 4 - 輸出 P4 特徵
        x = self.stage4(x)
        p4 = x

        # Stage 5 + SPPCSPC - 輸出 P5 特徵
        x = self.stage5(x)
        p5 = self.sppcspc(x)

        return p3, p4, p5

    def getOutputChannels(self) -> Dict[str, int]:
        """
        取得各層輸出的通道數

        返回:
            字典，包含 'p3', 'p4', 'p5' 的通道數
        """
        return {
            'p3': self.p3Channels,
            'p4': self.p4Channels,
            'p5': self.p5Channels
        }


class YOLOv7TinyBackbone(nn.Module):
    """
    YOLOv7-Tiny 輕量版骨幹網路

    這是 YOLOv7 骨幹網路的輕量版本，特點：
    - 更少的通道數
    - 更簡單的模組結構
    - 適合邊緣裝置和即時應用

    屬性:
        stem: 輸入處理層
        stages: 各階段特徵提取模組
    """

    def __init__(
        self,
        inChannels: int = 3,
        baseChannels: int = 32,
        widthMultiple: float = 1.0
    ):
        """
        初始化輕量版骨幹網路

        參數:
            inChannels: 輸入通道數
            baseChannels: 基礎通道數
            widthMultiple: 寬度倍數
        """
        super().__init__()

        def ch(channels: int) -> int:
            return max(int(channels * widthMultiple), 1)

        # Stem
        self.stem = nn.Sequential(
            Conv(inChannels, ch(32), 3, 2),    # H/2
            Conv(ch(32), ch(64), 3, 2)         # H/4
        )

        # Stage 2 - 簡化的特徵聚合
        self.stage2 = self._makeSimpleStage(ch(64), ch(64), ch(32))

        # Stage 3 (P3/8)
        self.stage3 = nn.Sequential(
            MPConv(ch(64)),
            self._makeSimpleBlock(ch(64), ch(128), ch(64))
        )
        self.p3Channels = ch(128)

        # Stage 4 (P4/16)
        self.stage4 = nn.Sequential(
            MPConv(ch(128)),
            self._makeSimpleBlock(ch(128), ch(256), ch(128))
        )
        self.p4Channels = ch(256)

        # Stage 5 (P5/32)
        self.stage5 = nn.Sequential(
            MPConv(ch(256)),
            self._makeSimpleBlock(ch(256), ch(512), ch(256))
        )
        self.p5Channels = ch(512)

    def _makeSimpleStage(self, inChannels: int, outChannels: int, hiddenChannels: int) -> nn.Module:
        """
        建立簡化的特徵階段

        參數:
            inChannels: 輸入通道數
            outChannels: 輸出通道數
            hiddenChannels: 中間通道數

        返回:
            特徵階段模組
        """
        return nn.Sequential(
            Conv(inChannels, hiddenChannels, 1, 1),
            Conv(hiddenChannels, hiddenChannels, 3, 1),
            Conv(hiddenChannels, hiddenChannels, 3, 1),
            Conv(hiddenChannels, outChannels, 1, 1)
        )

    def _makeSimpleBlock(self, inChannels: int, outChannels: int, hiddenChannels: int) -> nn.Module:
        """
        建立簡化的特徵區塊

        參數:
            inChannels: 輸入通道數
            outChannels: 輸出通道數
            hiddenChannels: 中間通道數

        返回:
            特徵區塊模組
        """
        class SimpleBlock(nn.Module):
            def __init__(self, inCh, outCh, hiddenCh):
                super().__init__()
                self.cv1 = Conv(inCh, hiddenCh, 1, 1)
                self.cv2 = Conv(inCh, hiddenCh, 1, 1)
                self.cv3 = Conv(hiddenCh, hiddenCh, 3, 1)
                self.cv4 = Conv(hiddenCh, hiddenCh, 3, 1)
                self.cv5 = Conv(hiddenCh * 4, outCh, 1, 1)

            def forward(self, x):
                branch1 = self.cv1(x)
                branch2 = self.cv2(x)
                branch3 = self.cv3(branch2)
                branch4 = self.cv4(branch3)
                return self.cv5(torch.cat([branch1, branch2, branch3, branch4], dim=1))

        return SimpleBlock(inChannels, outChannels, hiddenChannels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向傳播

        參數:
            x: 輸入影像張量

        返回:
            三個尺度的特徵圖 (p3, p4, p5)
        """
        x = self.stem(x)
        x = self.stage2(x)

        x = self.stage3(x)
        p3 = x

        x = self.stage4(x)
        p4 = x

        x = self.stage5(x)
        p5 = x

        return p3, p4, p5

    def getOutputChannels(self) -> Dict[str, int]:
        """取得各層輸出的通道數"""
        return {
            'p3': self.p3Channels,
            'p4': self.p4Channels,
            'p5': self.p5Channels
        }


def buildBackbone(config: Dict[str, Any]) -> nn.Module:
    """
    根據設定檔建立骨幹網路

    參數:
        config: 模型設定字典

    返回:
        骨幹網路模組
    """
    modelType = config.get('model', 'yolov7')
    depthMultiple = config.get('depthMultiple', 1.0)
    widthMultiple = config.get('widthMultiple', 1.0)

    if modelType.lower() == 'yolov7tiny':
        return YOLOv7TinyBackbone(widthMultiple=widthMultiple)
    else:
        return YOLOv7Backbone(
            depthMultiple=depthMultiple,
            widthMultiple=widthMultiple
        )
