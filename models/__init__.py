"""
模型模組 (Models Module)

此模組包含 YOLOv7 的所有神經網路架構元件：
- common: 通用模組 (Conv, Bottleneck, SPPCSPC 等)
- backbone: 骨幹網路 (E-ELAN)
- neck: 頸部網路 (FPN + PAN)
- head: 檢測頭
- yolo: 完整的 YOLO 模型

使用方式:
    from models import YOLOv7, buildModel, loadWeights
"""

from .yolo import YOLOv7, buildModel, loadWeights, loadYoloConfig
from .common import Conv, Bottleneck, SPPCSPC, ELANBlock, RepConv
from .backbone import YOLOv7Backbone, YOLOv7TinyBackbone
from .neck import FPNPAN, TinyFPNPAN
from .head import Detect, YOLOHead

__all__ = [
    # 主要模型
    'YOLOv7',
    'buildModel',
    'loadWeights',
    'loadYoloConfig',
    # 骨幹網路
    'YOLOv7Backbone',
    'YOLOv7TinyBackbone',
    # 頸部網路
    'FPNPAN',
    'TinyFPNPAN',
    # 檢測頭
    'Detect',
    'YOLOHead',
    # 通用模組
    'Conv',
    'Bottleneck',
    'SPPCSPC',
    'ELANBlock',
    'RepConv',
]
