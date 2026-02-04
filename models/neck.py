"""
頸部網路模組 (Neck Network Module)

頸部網路 (Neck) 位於骨幹網路和檢測頭之間，負責多尺度特徵融合。
YOLOv7 使用 FPN + PAN 結構:
- FPN (Feature Pyramid Network): 自頂向下傳遞語義資訊
- PAN (Path Aggregation Network): 自底向上傳遞定位資訊

為什麼需要特徵融合？
1. 低層特徵（如 P3）有豐富的空間細節，但語義資訊較弱
2. 高層特徵（如 P5）有豐富的語義資訊，但空間細節較弱
3. 透過特徵融合，每個尺度都能同時擁有語義和定位資訊

這對於檢測不同大小的物體至關重要：
- 小物體需要高解析度的特徵圖
- 大物體需要大感受野的特徵
"""

from typing import Tuple, Dict, Any

import torch
import torch.nn as nn

from .common import Conv, ELANBlock, Concat, Upsample, MPConv


class FPNPAN(nn.Module):
    """
    FPN + PAN 特徵融合網路

    結構示意圖:
    ```
                    骨幹網路輸出
                    ↓
    P5 ────────────→ SPPCSPC
                    ↓
                [FPN 自頂向下]
                    ↓
         ┌─────── P4 + 上採樣(P5')
         │         ↓
         │     融合後的 P4
         │         ↓
         │  ┌─── P3 + 上採樣(P4')
         │  │      ↓
         │  │  融合後的 P3  ←─────────────┐
         │  │                             │
         │  └──────────────────────────→[PAN 自底向上]
         │                                ↓
         │              P3' + 下採樣 ──→ P4''
         │                                ↓
         └───────────────────────────→ P4'' + 下採樣 ──→ P5''
                                                          ↓
                                                    最終輸出
    ```

    屬性:
        各種卷積和融合模組
    """

    def __init__(
        self,
        inChannels: Dict[str, int],
        outChannels: int = 256,
        widthMultiple: float = 1.0
    ):
        """
        初始化 FPN+PAN 網路

        參數:
            inChannels: 骨幹網路各層輸出的通道數
                       {'p3': 512, 'p4': 1024, 'p5': 512}
            outChannels: 輸出通道數（各尺度相同）
            widthMultiple: 寬度倍數
        """
        super().__init__()

        def ch(channels: int) -> int:
            return max(int(channels * widthMultiple), 1)

        # 取得輸入通道數
        p3Channels = inChannels['p3']
        p4Channels = inChannels['p4']
        p5Channels = inChannels['p5']

        # ============= FPN 自頂向下路徑 =============
        # P5 -> P4 融合

        # P5 降維
        self.fpnP5Conv = Conv(p5Channels, ch(256), 1, 1)

        # P5 上採樣
        self.fpnP5Upsample = Upsample(scaleFactor=2)

        # P4 側邊卷積（調整通道數以便拼接）
        self.fpnP4Lateral = Conv(p4Channels, ch(256), 1, 1)

        # P4 融合後的處理
        self.fpnP4Elan = ELANBlock(ch(512), ch(256), ch(64))

        # P4 -> P3 融合
        self.fpnP4Conv = Conv(ch(256), ch(128), 1, 1)
        self.fpnP4Upsample = Upsample(scaleFactor=2)
        self.fpnP3Lateral = Conv(p3Channels, ch(128), 1, 1)
        self.fpnP3Elan = ELANBlock(ch(256), ch(128), ch(32))

        # ============= PAN 自底向上路徑 =============
        # P3 -> P4 融合

        # P3 下採樣
        self.panP3Down = MPConv(ch(128))

        # P4 融合
        self.panP4Elan = ELANBlock(ch(256 + 128), ch(256), ch(64))

        # P4 -> P5 融合
        self.panP4Down = MPConv(ch(256))

        # P5 融合
        self.panP5Elan = ELANBlock(ch(512 + 256), ch(512), ch(128))

        # 記錄輸出通道數
        self.outChannelsP3 = ch(128)
        self.outChannelsP4 = ch(256)
        self.outChannelsP5 = ch(512)

        # 拼接模組
        self.concat = Concat(dimension=1)

    def forward(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向傳播

        參數:
            p3: 小物體特徵 [batch, p3_channels, H/8, W/8]
            p4: 中物體特徵 [batch, p4_channels, H/16, W/16]
            p5: 大物體特徵 [batch, p5_channels, H/32, W/32]

        返回:
            融合後的三尺度特徵 (out_p3, out_p4, out_p5)
        """
        # ============= FPN 自頂向下 =============

        # P5 -> P4
        fpnP5 = self.fpnP5Conv(p5)  # 降維
        fpnP5Up = self.fpnP5Upsample(fpnP5)  # 上採樣到 P4 尺寸
        fpnP4Lateral = self.fpnP4Lateral(p4)  # P4 側邊連接
        fpnP4 = self.fpnP4Elan(self.concat([fpnP4Lateral, fpnP5Up]))  # 融合

        # P4 -> P3
        fpnP4Conv = self.fpnP4Conv(fpnP4)  # 降維
        fpnP4Up = self.fpnP4Upsample(fpnP4Conv)  # 上採樣到 P3 尺寸
        fpnP3Lateral = self.fpnP3Lateral(p3)  # P3 側邊連接
        fpnP3 = self.fpnP3Elan(self.concat([fpnP3Lateral, fpnP4Up]))  # 融合

        # ============= PAN 自底向上 =============

        # P3 -> P4
        panP3Down = self.panP3Down(fpnP3)  # 下採樣
        panP4 = self.panP4Elan(self.concat([panP3Down, fpnP4]))  # 融合

        # P4 -> P5
        panP4Down = self.panP4Down(panP4)  # 下採樣
        panP5 = self.panP5Elan(self.concat([panP4Down, fpnP5]))  # 融合

        # 輸出
        outP3 = fpnP3   # 小物體特徵
        outP4 = panP4   # 中物體特徵
        outP5 = panP5   # 大物體特徵

        return outP3, outP4, outP5

    def getOutputChannels(self) -> Dict[str, int]:
        """
        取得輸出通道數

        返回:
            各尺度的輸出通道數
        """
        return {
            'p3': self.outChannelsP3,
            'p4': self.outChannelsP4,
            'p5': self.outChannelsP5
        }


class TinyFPNPAN(nn.Module):
    """
    輕量版 FPN + PAN 網路

    這是 YOLOv7-Tiny 使用的簡化特徵融合網路，
    使用更少的通道數和更簡單的融合模組。

    屬性:
        各種卷積和融合模組
    """

    def __init__(
        self,
        inChannels: Dict[str, int],
        widthMultiple: float = 1.0
    ):
        """
        初始化輕量版 FPN+PAN

        參數:
            inChannels: 骨幹網路各層輸出的通道數
            widthMultiple: 寬度倍數
        """
        super().__init__()

        def ch(channels: int) -> int:
            return max(int(channels * widthMultiple), 1)

        p3Ch = inChannels['p3']
        p4Ch = inChannels['p4']
        p5Ch = inChannels['p5']

        # FPN 自頂向下
        self.fpnP5Conv = Conv(p5Ch, ch(128), 1, 1)
        self.fpnP5Up = Upsample(2)
        self.fpnP4Lateral = Conv(p4Ch, ch(128), 1, 1)
        self.fpnP4Conv = Conv(ch(256), ch(64), 1, 1)

        self.fpnP4Conv2 = Conv(ch(64), ch(64), 1, 1)
        self.fpnP4Up = Upsample(2)
        self.fpnP3Lateral = Conv(p3Ch, ch(64), 1, 1)
        self.fpnP3Conv = Conv(ch(128), ch(64), 1, 1)

        # PAN 自底向上
        self.panP3Down = Conv(ch(64), ch(64), 3, 2)
        self.panP4Conv = Conv(ch(128), ch(128), 1, 1)

        self.panP4Down = Conv(ch(128), ch(128), 3, 2)
        self.panP5Conv = Conv(ch(256), ch(256), 1, 1)

        # 輸出通道數
        self.outChannelsP3 = ch(64)
        self.outChannelsP4 = ch(128)
        self.outChannelsP5 = ch(256)

        self.concat = Concat(1)

    def forward(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向傳播

        參數:
            p3, p4, p5: 骨幹網路的輸出特徵

        返回:
            融合後的特徵
        """
        # FPN
        fpnP5 = self.fpnP5Conv(p5)
        fpnP5Up = self.fpnP5Up(fpnP5)
        fpnP4 = self.fpnP4Conv(self.concat([self.fpnP4Lateral(p4), fpnP5Up]))

        fpnP4Conv = self.fpnP4Conv2(fpnP4)
        fpnP4Up = self.fpnP4Up(fpnP4Conv)
        fpnP3 = self.fpnP3Conv(self.concat([self.fpnP3Lateral(p3), fpnP4Up]))

        # PAN
        panP3Down = self.panP3Down(fpnP3)
        panP4 = self.panP4Conv(self.concat([panP3Down, fpnP4]))

        panP4Down = self.panP4Down(panP4)
        panP5 = self.panP5Conv(self.concat([panP4Down, fpnP5]))

        return fpnP3, panP4, panP5

    def getOutputChannels(self) -> Dict[str, int]:
        """取得輸出通道數"""
        return {
            'p3': self.outChannelsP3,
            'p4': self.outChannelsP4,
            'p5': self.outChannelsP5
        }


def buildNeck(config: Dict[str, Any], backboneChannels: Dict[str, int]) -> nn.Module:
    """
    根據設定檔建立頸部網路

    參數:
        config: 模型設定字典
        backboneChannels: 骨幹網路的輸出通道數

    返回:
        頸部網路模組
    """
    modelType = config.get('model', 'yolov7')
    widthMultiple = config.get('widthMultiple', 1.0)

    if modelType.lower() == 'yolov7tiny':
        return TinyFPNPAN(
            inChannels=backboneChannels,
            widthMultiple=widthMultiple
        )
    else:
        return FPNPAN(
            inChannels=backboneChannels,
            widthMultiple=widthMultiple
        )
