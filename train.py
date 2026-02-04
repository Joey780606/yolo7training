"""
YOLOv7 訓練腳本 (Training Script)

此腳本實現了完整的 YOLOv7 訓練流程，包括：
- 資料載入和增強
- 模型建立和初始化
- 訓練循環（前向傳播、反向傳播、優化器更新）
- 驗證和評估
- 模型儲存和日誌記錄

使用方式:
    # 基本訓練
    python train.py --data configs/data/custom.yaml --epochs 100

    # 指定批次大小和影像大小
    python train.py --data configs/data/custom.yaml --epochs 100 --batchSize 16 --imgSize 640

    # 從預訓練權重繼續訓練
    python train.py --data configs/data/custom.yaml --weights weights/yolov7.pt --epochs 50
"""

import argparse
import os
import sys
import math
import random
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# 添加專案根目錄到路徑
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.yolo import YOLOv7, buildModel, loadWeights
from utils.datasets import createDataLoader, loadDataConfig
from utils.loss import ComputeLoss
from utils.metrics import MetricsCalculator, fitnessScore
from utils.general import (
    setRandomSeed, makeDirectory, incrementPath,
    nonMaxSuppression, scaleBoxes
)
from utils.torchUtils import (
    selectDevice, modelInfo, ModelEMA,
    getCosineScheduler, saveCheckpoint, loadCheckpoint
)


def parseArgs():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='YOLOv7 訓練程式',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 資料設定
    parser.add_argument('--data', type=str, default='configs/data/custom.yaml',
                       help='資料集設定檔路徑')
    parser.add_argument('--imgSize', type=int, default=640,
                       help='訓練影像大小')

    # 模型設定
    parser.add_argument('--model', type=str, default='yolov7',
                       choices=['yolov7', 'yolov7tiny'],
                       help='模型類型')
    parser.add_argument('--weights', type=str, default='',
                       help='預訓練權重路徑')

    # 訓練設定
    parser.add_argument('--epochs', type=int, default=100,
                       help='訓練週期數')
    parser.add_argument('--batchSize', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--workers', type=int, default=8,
                       help='資料載入工作執行緒數')

    # 優化器設定
    parser.add_argument('--lr', type=float, default=0.01,
                       help='初始學習率')
    parser.add_argument('--momentum', type=float, default=0.937,
                       help='SGD 動量')
    parser.add_argument('--weightDecay', type=float, default=0.0005,
                       help='權重衰減')

    # 其他設定
    parser.add_argument('--device', type=str, default='',
                       help='計算裝置 (cuda:0, cpu 等)')
    parser.add_argument('--seed', type=int, default=42,
                       help='隨機種子')
    parser.add_argument('--project', type=str, default='runs/train',
                       help='訓練輸出目錄')
    parser.add_argument('--name', type=str, default='exp',
                       help='實驗名稱')
    parser.add_argument('--existOk', action='store_true',
                       help='是否允許覆蓋已存在的實驗目錄')
    parser.add_argument('--resume', type=str, default='',
                       help='從檢查點繼續訓練')

    # 進階設定
    parser.add_argument('--noAugment', action='store_true',
                       help='禁用資料增強')
    parser.add_argument('--noEma', action='store_true',
                       help='禁用 EMA')
    parser.add_argument('--noAmp', action='store_true',
                       help='禁用混合精度訓練')

    return parser.parse_args()


def getHyperParams() -> Dict:
    """
    取得訓練超參數

    這些超參數控制資料增強、損失權重等。
    可以根據資料集特性進行調整。

    返回:
        超參數字典
    """
    return {
        # 學習率相關
        'lr0': 0.01,           # 初始學習率
        'lrf': 0.01,           # 最終學習率比例 (相對於 lr0)
        'momentum': 0.937,     # SGD 動量
        'weight_decay': 0.0005, # 權重衰減

        # 預熱相關
        'warmup_epochs': 3,    # 預熱週期數
        'warmup_momentum': 0.8, # 預熱期間的初始動量
        'warmup_bias_lr': 0.1, # 預熱期間偏置的學習率

        # 損失權重
        'box': 0.05,           # 邊界框損失權重
        'cls': 0.3,            # 分類損失權重
        'obj': 0.7,            # 物件性損失權重

        # 資料增強
        'hsv_h': 0.015,        # HSV 色調增強範圍
        'hsv_s': 0.7,          # HSV 飽和度增強範圍
        'hsv_v': 0.4,          # HSV 明度增強範圍
        'degrees': 0.0,        # 旋轉角度範圍
        'translate': 0.1,      # 平移範圍
        'scale': 0.5,          # 縮放範圍
        'shear': 0.0,          # 剪切角度範圍
        'perspective': 0.0,    # 透視變換範圍
        'flipud': 0.0,         # 垂直翻轉機率
        'fliplr': 0.5,         # 水平翻轉機率
        'mosaic': 1.0,         # Mosaic 增強機率
        'mixup': 0.0,          # MixUp 增強機率

        # 其他
        'anchor_t': 4.0,       # 錨點框匹配閾值
        'label_smoothing': 0.0, # 標籤平滑係數
    }


def trainOneEpoch(
    model: nn.Module,
    dataLoader,
    optimizer: optim.Optimizer,
    computeLoss: ComputeLoss,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    totalEpochs: int,
    ema: Optional[ModelEMA] = None,
    useAmp: bool = True
) -> Dict[str, float]:
    """
    訓練一個週期

    參數:
        model: 模型
        dataLoader: 訓練資料載入器
        optimizer: 優化器
        computeLoss: 損失計算器
        scaler: 混合精度縮放器
        device: 計算裝置
        epoch: 當前週期
        totalEpochs: 總週期數
        ema: EMA 模型
        useAmp: 是否使用混合精度

    返回:
        損失字典
    """
    model.train()

    # 初始化累積損失
    totalLoss = 0.0
    batchLosses = {'box': 0.0, 'obj': 0.0, 'cls': 0.0}

    # 進度條
    pbar = tqdm(dataLoader, desc=f'訓練 Epoch {epoch + 1}/{totalEpochs}')

    for batchIdx, (images, targets, paths, shapes) in enumerate(pbar):
        # 將資料移動到裝置
        images = images.to(device, non_blocking=True)
        targets = targets.to(device)

        # 前向傳播（使用混合精度）
        with autocast(enabled=useAmp):
            predictions = model(images)
            loss, lossItems = computeLoss(predictions, targets)

        # 反向傳播
        scaler.scale(loss).backward()

        # 更新參數
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # 更新 EMA
        if ema is not None:
            ema.update(model)

        # 累積損失
        totalLoss += loss.item()
        for key in batchLosses:
            batchLosses[key] += lossItems.get(key, 0.0)

        # 更新進度條
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'box': f'{lossItems["box"]:.4f}',
            'obj': f'{lossItems["obj"]:.4f}',
            'cls': f'{lossItems["cls"]:.4f}'
        })

    # 計算平均損失
    numBatches = len(dataLoader)
    avgLosses = {
        'total': totalLoss / numBatches,
        'box': batchLosses['box'] / numBatches,
        'obj': batchLosses['obj'] / numBatches,
        'cls': batchLosses['cls'] / numBatches
    }

    return avgLosses


@torch.no_grad()
def validate(
    model: nn.Module,
    dataLoader,
    device: torch.device,
    numClasses: int,
    confThreshold: float = 0.001,
    iouThreshold: float = 0.6
) -> Dict[str, float]:
    """
    驗證模型

    參數:
        model: 模型
        dataLoader: 驗證資料載入器
        device: 計算裝置
        numClasses: 類別數量
        confThreshold: 信心閾值
        iouThreshold: NMS IoU 閾值

    返回:
        指標字典
    """
    model.eval()

    # 初始化指標計算器
    metricsCalc = MetricsCalculator(numClasses)

    pbar = tqdm(dataLoader, desc='驗證中')

    for images, targets, paths, shapes in pbar:
        images = images.to(device, non_blocking=True)

        # 推論
        predictions, _ = model(images)

        # NMS
        predictions = nonMaxSuppression(
            predictions,
            confThreshold=confThreshold,
            iouThreshold=iouThreshold
        )

        # 處理每張圖片
        for i, pred in enumerate(predictions):
            # 取得這張圖片的標籤
            imageLables = targets[targets[:, 0] == i]

            if pred is not None and len(pred) > 0:
                pred = pred.cpu().numpy()
            else:
                pred = np.zeros((0, 6))

            if len(imageLables) > 0:
                # 轉換標籤格式
                labels = imageLables[:, 1:].cpu().numpy()
            else:
                labels = np.zeros((0, 5))

            # 更新指標
            metricsCalc.update(pred, labels)

    # 計算最終指標
    metrics = metricsCalc.compute()

    return metrics


def main():
    """主函數"""
    args = parseArgs()

    # 設定隨機種子
    setRandomSeed(args.seed)

    # 選擇裝置
    device = selectDevice(args.device, args.batchSize)

    # 建立輸出目錄
    saveDir = incrementPath(Path(args.project) / args.name, existOk=args.existOk)
    weightsDir = saveDir / 'weights'
    weightsDir.mkdir(parents=True, exist_ok=True)

    print(f'\n訓練結果將儲存到: {saveDir}')

    # ============= 載入設定 =============
    print('\n載入設定...')

    # 載入資料設定
    try:
        dataConfig = loadDataConfig(args.data)
    except FileNotFoundError:
        print(f'錯誤: 找不到資料設定檔 {args.data}')
        sys.exit(1)
    except Exception as e:
        print(f'錯誤: 載入資料設定檔時發生錯誤: {e}')
        sys.exit(1)

    numClasses = dataConfig.get('nc', 80)
    classNames = dataConfig.get('names', [f'class_{i}' for i in range(numClasses)])

    print(f'類別數量: {numClasses}')

    # 取得超參數
    hyperParams = getHyperParams()

    # ============= 建立資料載入器 =============
    print('\n建立資料載入器...')

    trainPath = dataConfig.get('train')
    valPath = dataConfig.get('val')

    if not trainPath:
        print('錯誤: 資料設定檔中沒有指定訓練集路徑')
        sys.exit(1)

    try:
        trainLoader, trainDataset = createDataLoader(
            path=trainPath,
            imageSize=args.imgSize,
            batchSize=args.batchSize,
            hyperParams=hyperParams,
            augment=not args.noAugment,
            shuffle=True,
            numWorkers=args.workers,
            prefix='訓練: '
        )
    except Exception as e:
        print(f'錯誤: 建立訓練資料載入器時發生錯誤: {e}')
        sys.exit(1)

    valLoader = None
    if valPath:
        try:
            valLoader, valDataset = createDataLoader(
                path=valPath,
                imageSize=args.imgSize,
                batchSize=args.batchSize * 2,  # 驗證時可以用更大的批次
                hyperParams=hyperParams,
                augment=False,
                shuffle=False,
                numWorkers=args.workers,
                prefix='驗證: '
            )
        except Exception as e:
            print(f'警告: 建立驗證資料載入器時發生錯誤: {e}')

    # ============= 建立模型 =============
    print('\n建立模型...')

    model = YOLOv7(
        numClasses=numClasses,
        modelType=args.model
    )

    # 載入預訓練權重
    if args.weights:
        try:
            model = loadWeights(model, args.weights, device=device, strict=False)
        except FileNotFoundError:
            print(f'警告: 找不到權重檔案 {args.weights}')
        except Exception as e:
            print(f'警告: 載入權重時發生錯誤: {e}')

    model = model.to(device)
    model.info()

    # ============= 設定優化器 =============
    print('\n設定優化器...')

    # 分組參數（不同組使用不同學習率）
    params = []
    bnParams = []  # BatchNorm 參數
    biasParams = []  # 偏置參數
    otherParams = []  # 其他參數

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name:
            biasParams.append(param)
        elif 'bn' in name or 'BatchNorm' in name:
            bnParams.append(param)
        else:
            otherParams.append(param)

    # 建立參數組
    paramGroups = [
        {'params': otherParams, 'lr': args.lr},
        {'params': bnParams, 'lr': args.lr, 'weight_decay': 0.0},
        {'params': biasParams, 'lr': args.lr, 'weight_decay': 0.0}
    ]

    optimizer = optim.SGD(
        paramGroups,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weightDecay,
        nesterov=True
    )

    # 學習率排程器
    scheduler = getCosineScheduler(
        optimizer,
        totalEpochs=args.epochs,
        warmupEpochs=hyperParams['warmup_epochs']
    )

    # ============= 其他訓練元件 =============

    # 損失計算器
    computeLoss = ComputeLoss(model, hyperParams)

    # EMA
    ema = None
    if not args.noEma:
        ema = ModelEMA(model)

    # 混合精度
    scaler = GradScaler(enabled=not args.noAmp and device.type == 'cuda')

    # 繼續訓練
    startEpoch = 0
    bestFitness = 0.0

    if args.resume:
        try:
            checkpoint = loadCheckpoint(model, args.resume, optimizer, ema, device)
            startEpoch = checkpoint.get('epoch', 0) + 1
            bestFitness = checkpoint.get('bestFitness', 0.0)
            print(f'從週期 {startEpoch} 繼續訓練')
        except Exception as e:
            print(f'警告: 無法從檢查點繼續訓練: {e}')

    # ============= 訓練循環 =============
    print(f'\n開始訓練 {args.epochs} 個週期...\n')

    for epoch in range(startEpoch, args.epochs):
        # 訓練一個週期
        trainLosses = trainOneEpoch(
            model=model,
            dataLoader=trainLoader,
            optimizer=optimizer,
            computeLoss=computeLoss,
            scaler=scaler,
            device=device,
            epoch=epoch,
            totalEpochs=args.epochs,
            ema=ema,
            useAmp=not args.noAmp
        )

        # 更新學習率
        scheduler.step()
        currentLr = optimizer.param_groups[0]['lr']

        # 驗證
        metrics = {'mAP50': 0.0, 'mAP50_95': 0.0}
        if valLoader is not None:
            evalModel = ema.ema if ema else model
            metrics = validate(
                model=evalModel,
                dataLoader=valLoader,
                device=device,
                numClasses=numClasses
            )

        # 計算適應度
        fitness = fitnessScore(metrics)

        # 列印結果
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        print(f'  訓練損失: box={trainLosses["box"]:.4f}, obj={trainLosses["obj"]:.4f}, cls={trainLosses["cls"]:.4f}')
        print(f'  驗證指標: mAP@0.5={metrics["mAP50"]:.4f}, mAP@0.5:0.95={metrics["mAP50_95"]:.4f}')
        print(f'  學習率: {currentLr:.6f}')

        # 儲存檢查點
        isBest = fitness > bestFitness
        if isBest:
            bestFitness = fitness

        # 儲存最後和最佳模型
        saveModel = ema.ema if ema else model

        # 儲存最後的檢查點
        saveCheckpoint(
            model=saveModel,
            optimizer=optimizer,
            epoch=epoch,
            bestFitness=bestFitness,
            savePath=str(weightsDir / 'last.pt'),
            ema=ema
        )

        # 儲存最佳模型
        if isBest:
            saveCheckpoint(
                model=saveModel,
                optimizer=optimizer,
                epoch=epoch,
                bestFitness=bestFitness,
                savePath=str(weightsDir / 'best.pt'),
                ema=ema
            )
            print(f'  * 新的最佳模型! (fitness={fitness:.4f})')

        print()

    # ============= 訓練完成 =============
    print('='*50)
    print('訓練完成!')
    print(f'最佳適應度: {bestFitness:.4f}')
    print(f'結果儲存在: {saveDir}')
    print(f'  - 最後權重: {weightsDir}/last.pt')
    print(f'  - 最佳權重: {weightsDir}/best.pt')
    print('='*50)


if __name__ == '__main__':
    main()
