"""
PyTorch 輔助函式模組 (PyTorch Utilities Module)

此模組提供 PyTorch 相關的輔助函式，包括：
- 裝置選擇（自動偵測 GPU/CPU）
- 模型權重初始化
- 模型資訊顯示
- 梯度操作工具
- 學習率排程器

這些工具函式簡化了深度學習訓練流程中的常見操作。
"""

import math
import os
from typing import List, Optional, Tuple, Dict, Any
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def selectDevice(device: str = '', batchSize: int = 0) -> torch.device:
    """
    自動選擇最佳計算裝置 (GPU 優先，否則使用 CPU)

    這個函式會自動偵測可用的硬體，並選擇最適合的計算裝置。
    如果有 CUDA GPU 可用，會優先使用；否則回退到 CPU。

    參數:
        device: 指定裝置，可以是 'cpu', 'cuda', 'cuda:0', 'cuda:1' 等
                如果為空字串，則自動選擇
        batchSize: 批次大小，用於多 GPU 時的驗證

    返回:
        選定的 torch.device 物件
    """
    # 如果用戶指定了 CPU
    if device.lower() == 'cpu':
        print('使用裝置: CPU')
        return torch.device('cpu')

    # 檢查 CUDA 是否可用
    cudaAvailable = torch.cuda.is_available()

    if not cudaAvailable:
        print('警告: CUDA 不可用，將使用 CPU 進行運算')
        print('提示: 使用 GPU 可以大幅加速訓練過程')
        return torch.device('cpu')

    # 如果用戶指定了特定的 CUDA 裝置
    if device:
        # 設定 CUDA 可見裝置
        os.environ['CUDA_VISIBLE_DEVICES'] = device.replace('cuda:', '')
        deviceObj = torch.device(device)
    else:
        # 自動選擇第一個可用的 GPU
        deviceObj = torch.device('cuda:0')

    # 顯示 GPU 資訊
    numGpus = torch.cuda.device_count()
    if numGpus > 1:
        print(f'偵測到 {numGpus} 個 GPU')

    for i in range(numGpus):
        props = torch.cuda.get_device_properties(i)
        memoryGb = props.total_memory / (1024 ** 3)
        print(f'  GPU {i}: {props.name} ({memoryGb:.1f} GB)')

    print(f'使用裝置: {deviceObj}')

    # 驗證批次大小（多 GPU 時需要能被 GPU 數量整除）
    if batchSize > 0 and numGpus > 1 and batchSize % numGpus != 0:
        print(f'警告: 批次大小 {batchSize} 不能被 GPU 數量 {numGpus} 整除')
        print(f'建議: 使用 {batchSize - batchSize % numGpus} 或 {batchSize + numGpus - batchSize % numGpus}')

    return deviceObj


def countParameters(model: nn.Module) -> int:
    """
    計算模型的參數總數

    參數:
        model: PyTorch 模型

    返回:
        可訓練參數的總數
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def modelInfo(model: nn.Module, verbose: bool = False) -> Tuple[int, int, int]:
    """
    顯示模型資訊

    包括：模型層數、參數總數、梯度數量等。
    這對於了解模型架構和確認模型正確載入很有幫助。

    參數:
        model: PyTorch 模型
        verbose: 是否顯示每層的詳細資訊

    返回:
        (層數, 總參數數, 可訓練參數數)
    """
    # 計算層數
    numLayers = len(list(model.modules()))

    # 計算參數數量
    numParams = sum(p.numel() for p in model.parameters())
    numGrads = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f'{"層名稱":<40} {"輸入形狀":<20} {"輸出形狀":<20} {"參數數量":<15}')
        print('-' * 95)
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and hasattr(module.weight, 'shape'):
                params = module.weight.numel()
                if hasattr(module, 'bias') and module.bias is not None:
                    params += module.bias.numel()
                print(f'{name:<40} {str(module.weight.shape):<20} {"-":<20} {params:<15}')

    print(f'\n模型摘要:')
    print(f'  層數: {numLayers}')
    print(f'  參數總數: {numParams:,}')
    print(f'  可訓練參數: {numGrads:,}')
    print(f'  模型大小: {numParams * 4 / (1024 ** 2):.2f} MB (FP32)')

    return numLayers, numParams, numGrads


def initializeWeights(model: nn.Module) -> None:
    """
    初始化模型權重

    使用適當的初始化方法可以加速訓練收斂並提高模型性能。
    不同類型的層使用不同的初始化策略：
    - 卷積層: Kaiming 初始化（適合 ReLU 激活函數）
    - 批次正規化: 權重設為 1，偏置設為 0
    - 線性層: 正態分布初始化

    參數:
        model: 要初始化的 PyTorch 模型
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # Kaiming 初始化，適合使用 ReLU/LeakyReLU 的網路
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.BatchNorm2d):
            # BatchNorm: γ=1, β=0
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.Linear):
            # 線性層使用正態分布
            nn.init.normal_(module.weight, 0, 0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


def fuseConvAndBn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    融合卷積層和批次正規化層

    在推論時，可以將卷積層和後續的批次正規化層融合成單一卷積層。
    這可以減少運算量並加速推論，同時保持數學等價性。

    數學原理:
    BatchNorm: y = γ * (x - μ) / σ + β
    Conv: y = W * x + b
    融合後: y = (γ/σ * W) * x + (γ/σ * (b - μ) + β)

    參數:
        conv: 卷積層
        bn: 批次正規化層

    返回:
        融合後的卷積層
    """
    # 建立新的卷積層（帶有偏置）
    fusedConv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True
    ).requires_grad_(False).to(conv.weight.device)

    # 準備權重
    wConv = conv.weight.clone().view(conv.out_channels, -1)
    wBn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))

    # 融合權重: W_fused = γ/σ * W_conv
    fusedConv.weight.copy_(torch.mm(wBn, wConv).view(fusedConv.weight.shape))

    # 融合偏置: b_fused = γ/σ * (b_conv - μ) + β
    bConv = conv.bias if conv.bias is not None else torch.zeros(conv.weight.size(0), device=conv.weight.device)
    bBn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedConv.bias.copy_(torch.mm(wBn, bConv.reshape(-1, 1)).reshape(-1) + bBn)

    return fusedConv


def copyAttr(target: object, source: object, includes: List[str] = [], excludes: List[str] = []) -> None:
    """
    從來源物件複製屬性到目標物件

    參數:
        target: 目標物件
        source: 來源物件
        includes: 要包含的屬性列表（如果為空則複製所有）
        excludes: 要排除的屬性列表
    """
    for key, value in source.__dict__.items():
        if (not includes or key in includes) and key not in excludes:
            setattr(target, key, value)


class ModelEMA:
    """
    模型指數移動平均 (Exponential Moving Average)

    EMA 是一種訓練技巧，維護模型參數的移動平均值。
    這可以讓模型更穩定，並在某些情況下提高最終性能。

    EMA 的更新公式:
    θ_ema = decay * θ_ema + (1 - decay) * θ_model

    其中 decay 通常設為 0.9999，這意味著 EMA 參數會緩慢地跟隨模型參數。

    屬性:
        ema: EMA 模型副本
        updates: 更新次數
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: float = 2000):
        """
        初始化 EMA

        參數:
            model: 原始模型
            decay: 衰減率
            tau: 預熱期的時間常數
        """
        # 建立 EMA 模型（深拷貝）
        self.ema = deepcopy(model).eval()

        # 凍結 EMA 模型的參數
        for param in self.ema.parameters():
            param.requires_grad_(False)

        self.updates = 0
        self.decay = decay
        self.tau = tau

    def update(self, model: nn.Module) -> None:
        """
        更新 EMA 參數

        參數:
            model: 當前訓練中的模型
        """
        self.updates += 1

        # 計算當前的衰減率（考慮預熱期）
        # 在訓練初期，衰減率較低，讓 EMA 更快地跟上模型
        decay = self.decay * (1 - math.exp(-self.updates / self.tau))

        # 更新 EMA 參數
        modelStateDict = model.state_dict()
        for key, emaParam in self.ema.state_dict().items():
            if emaParam.dtype.is_floating_point:
                emaParam *= decay
                emaParam += (1 - decay) * modelStateDict[key].detach()

    def updateAttr(self, model: nn.Module, includes: List[str] = [], excludes: List[str] = ['process_group', 'reducer']) -> None:
        """
        更新 EMA 模型的屬性

        參數:
            model: 來源模型
            includes: 要包含的屬性
            excludes: 要排除的屬性
        """
        copyAttr(self.ema, model, includes, excludes)


def deNormalize(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """
    反正規化張量

    將經過正規化的張量還原為原始範圍，用於視覺化。

    參數:
        tensor: 正規化後的張量
        mean: 正規化使用的平均值
        std: 正規化使用的標準差

    返回:
        反正規化後的張量
    """
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    return tensor * std + mean


def getOneCycleLr(
    optimizer: Optimizer,
    maxLr: float,
    totalSteps: int,
    pctStart: float = 0.3,
    divFactor: float = 25.0,
    finalDivFactor: float = 10000.0
) -> LambdaLR:
    """
    建立 One Cycle 學習率排程器

    One Cycle 學習率策略是一種有效的訓練技巧：
    1. 從低學習率開始，逐漸增加到最大值（預熱階段）
    2. 從最大值逐漸降低到很小的值（退火階段）

    這種策略可以讓模型在訓練初期穩定學習，中期快速收斂，
    後期精細調整。

    參數:
        optimizer: 優化器
        maxLr: 最大學習率
        totalSteps: 總訓練步數
        pctStart: 預熱階段佔總步數的比例
        divFactor: 初始學習率 = maxLr / divFactor
        finalDivFactor: 最終學習率 = maxLr / finalDivFactor

    返回:
        LambdaLR 排程器
    """
    def lrLambda(step: int) -> float:
        if step < totalSteps * pctStart:
            # 預熱階段：線性增加
            return (1 - 1 / divFactor) * step / (totalSteps * pctStart) + 1 / divFactor
        else:
            # 退火階段：餘弦退火
            progress = (step - totalSteps * pctStart) / (totalSteps * (1 - pctStart))
            return (1 - 1 / finalDivFactor) * 0.5 * (1 + math.cos(math.pi * progress)) + 1 / finalDivFactor

    return LambdaLR(optimizer, lr_lambda=lrLambda)


def getCosineScheduler(
    optimizer: Optimizer,
    totalEpochs: int,
    warmupEpochs: int = 3,
    minLrRatio: float = 0.01
) -> LambdaLR:
    """
    建立餘弦退火學習率排程器（帶預熱）

    餘弦退火是一種平滑降低學習率的方法，學習率按照餘弦函數下降。
    加入預熱階段可以讓訓練初期更穩定。

    參數:
        optimizer: 優化器
        totalEpochs: 總訓練週期數
        warmupEpochs: 預熱週期數
        minLrRatio: 最小學習率相對於初始學習率的比例

    返回:
        LambdaLR 排程器
    """
    def lrLambda(epoch: int) -> float:
        if epoch < warmupEpochs:
            # 預熱階段：線性增加
            return (epoch + 1) / warmupEpochs
        else:
            # 餘弦退火階段
            progress = (epoch - warmupEpochs) / (totalEpochs - warmupEpochs)
            return minLrRatio + (1 - minLrRatio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lrLambda)


def saveCheckpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    bestFitness: float,
    savePath: str,
    ema: Optional[ModelEMA] = None
) -> None:
    """
    儲存訓練檢查點

    檢查點包含完整的訓練狀態，可以用於恢復訓練或提取最佳模型。

    參數:
        model: 模型
        optimizer: 優化器
        epoch: 當前週期
        bestFitness: 最佳性能指標
        savePath: 儲存路徑
        ema: EMA 模型（可選）
    """
    checkpoint = {
        'epoch': epoch,
        'bestFitness': bestFitness,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    if ema is not None:
        checkpoint['ema'] = ema.ema.state_dict()

    try:
        torch.save(checkpoint, savePath)
    except Exception as e:
        raise RuntimeError(f'儲存檢查點時發生錯誤: {e}')


def loadCheckpoint(
    model: nn.Module,
    checkpointPath: str,
    optimizer: Optional[Optimizer] = None,
    ema: Optional[ModelEMA] = None,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    載入訓練檢查點

    參數:
        model: 模型
        checkpointPath: 檢查點路徑
        optimizer: 優化器（可選，如果要恢復訓練）
        ema: EMA 模型（可選）
        device: 載入到的裝置

    返回:
        檢查點資訊字典
    """
    try:
        checkpoint = torch.load(checkpointPath, map_location=device)
    except FileNotFoundError:
        raise FileNotFoundError(f'找不到檢查點檔案: {checkpointPath}')
    except Exception as e:
        raise RuntimeError(f'載入檢查點時發生錯誤: {e}')

    # 載入模型權重
    model.load_state_dict(checkpoint['model'])

    # 載入優化器狀態（如果提供）
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # 載入 EMA 狀態（如果提供）
    if ema is not None and 'ema' in checkpoint:
        ema.ema.load_state_dict(checkpoint['ema'])

    return checkpoint


def autoMixedPrecision(enabled: bool = True) -> torch.cuda.amp.GradScaler:
    """
    設定自動混合精度訓練

    混合精度訓練使用 FP16 和 FP32 混合運算，可以：
    - 減少 GPU 記憶體使用量（約 50%）
    - 加速訓練（現代 GPU 對 FP16 有硬體加速）
    - 通過損失縮放保持訓練穩定性

    參數:
        enabled: 是否啟用混合精度

    返回:
        GradScaler 物件，用於損失縮放
    """
    return torch.cuda.amp.GradScaler(enabled=enabled)


def synchronizeCuda() -> None:
    """
    同步 CUDA 操作

    在計時或需要確保 GPU 操作完成時使用。
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def getGpuMemoryUsage(device: int = 0) -> Tuple[float, float]:
    """
    取得 GPU 記憶體使用情況

    參數:
        device: GPU 裝置編號

    返回:
        (已使用記憶體 GB, 總記憶體 GB)
    """
    if not torch.cuda.is_available():
        return (0.0, 0.0)

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)

    return (allocated, total)
