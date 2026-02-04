"""
YOLOv7 驗證腳本 (Validation Script)

此腳本用於評估訓練好的模型在驗證集上的性能。

輸出指標:
- Precision (精確率): 預測正確的比例
- Recall (召回率): 找到的正確物體比例
- mAP@0.5: IoU=0.5 時的平均精確度
- mAP@0.5:0.95: IoU 從 0.5 到 0.95 的平均精確度

使用方式:
    # 基本驗證
    python validate.py --weights runs/train/exp/weights/best.pt --data configs/data/custom.yaml

    # 調整閾值
    python validate.py --weights runs/train/exp/weights/best.pt --data configs/data/custom.yaml --confThreshold 0.001 --iouThreshold 0.6
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

# 添加專案根目錄到路徑
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.yolo import YOLOv7, loadWeights
from utils.datasets import createDataLoader, loadDataConfig
from utils.general import nonMaxSuppression, scaleBoxes, makeDirectory, xywh2xyxy
from utils.metrics import MetricsCalculator, ConfusionMatrix, apPerClass, fitnessScore
from utils.torchUtils import selectDevice


def parseArgs():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='YOLOv7 模型驗證程式',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 基本設定
    parser.add_argument('--weights', type=str, required=True,
                       help='模型權重路徑')
    parser.add_argument('--data', type=str, required=True,
                       help='資料集設定檔路徑')

    # 模型設定
    parser.add_argument('--imgSize', type=int, default=640,
                       help='推論影像大小')
    parser.add_argument('--model', type=str, default='yolov7',
                       choices=['yolov7', 'yolov7tiny'],
                       help='模型類型')
    parser.add_argument('--batchSize', type=int, default=32,
                       help='批次大小')

    # 驗證設定
    parser.add_argument('--confThreshold', type=float, default=0.001,
                       help='信心閾值')
    parser.add_argument('--iouThreshold', type=float, default=0.6,
                       help='NMS IoU 閾值')
    parser.add_argument('--device', type=str, default='',
                       help='計算裝置')
    parser.add_argument('--workers', type=int, default=8,
                       help='資料載入工作執行緒數')

    # 輸出設定
    parser.add_argument('--output', type=str, default='runs/validate',
                       help='輸出目錄')
    parser.add_argument('--saveTxt', action='store_true',
                       help='儲存預測結果為文字檔')
    parser.add_argument('--saveJson', action='store_true',
                       help='儲存預測結果為 COCO JSON 格式')
    parser.add_argument('--verbose', action='store_true',
                       help='顯示每個類別的詳細結果')

    return parser.parse_args()


@torch.no_grad()
def runValidation(
    model: torch.nn.Module,
    dataLoader,
    device: torch.device,
    numClasses: int,
    classNames: List[str],
    confThreshold: float = 0.001,
    iouThreshold: float = 0.6,
    saveTxt: bool = False,
    saveJson: bool = False,
    outputDir: str = '',
    verbose: bool = False
) -> Dict[str, float]:
    """
    執行驗證

    參數:
        model: 模型
        dataLoader: 驗證資料載入器
        device: 計算裝置
        numClasses: 類別數量
        classNames: 類別名稱
        confThreshold: 信心閾值
        iouThreshold: NMS IoU 閾值
        saveTxt: 是否儲存文字結果
        saveJson: 是否儲存 JSON 結果
        outputDir: 輸出目錄
        verbose: 是否顯示詳細資訊

    返回:
        指標字典
    """
    model.eval()

    # 初始化
    seen = 0  # 處理的影像數量
    confusionMatrix = ConfusionMatrix(numClasses, confThreshold, iouThreshold)

    # 儲存用於計算 AP 的資料
    stats = []  # [(correctAtIoUs, conf, predCls, targetCls), ...]
    jsonPredictions = []  # COCO 格式的預測

    # IoU 閾值（用於計算 mAP@0.5:0.95）
    iouThresholds = torch.linspace(0.5, 0.95, 10).to(device)
    numIouThresholds = iouThresholds.numel()

    # 標籤目錄
    if saveTxt:
        labelsDir = Path(outputDir) / 'labels'
        labelsDir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(dataLoader, desc='驗證中')

    for batchIdx, (images, targets, paths, shapes) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device)
        batchSize = images.shape[0]

        # 推論
        predictions, _ = model(images)

        # NMS
        predictions = nonMaxSuppression(
            predictions,
            confThreshold=confThreshold,
            iouThreshold=iouThreshold
        )

        # 處理每張圖片
        for idx in range(batchSize):
            seen += 1
            pred = predictions[idx]
            imagePath = paths[idx]
            shape = shapes[idx]

            # 取得這張圖片的標籤
            targetMask = targets[:, 0] == idx
            imageLabels = targets[targetMask]
            numLabels = len(imageLabels)

            # 目標類別
            if numLabels > 0:
                targetCls = imageLabels[:, 1].long()
                # 轉換標籤座標 (正規化的 xywh -> 絕對的 xyxy)
                targetBoxes = imageLabels[:, 2:6].clone()
                targetBoxes[:, 0] *= shape[1]  # x
                targetBoxes[:, 1] *= shape[0]  # y
                targetBoxes[:, 2] *= shape[1]  # w
                targetBoxes[:, 3] *= shape[0]  # h
                # xywh -> xyxy
                targetBoxes = xywh2xyxyTorch(targetBoxes)
            else:
                targetCls = torch.tensor([], device=device, dtype=torch.long)
                targetBoxes = torch.tensor([], device=device)

            # 處理預測
            if pred is None or len(pred) == 0:
                if numLabels > 0:
                    # 沒有預測但有標籤 -> 全部是漏檢
                    stats.append((
                        torch.zeros(0, numIouThresholds, device=device, dtype=torch.bool),
                        torch.tensor([], device=device),
                        torch.tensor([], device=device),
                        targetCls
                    ))
                continue

            # 預測的邊界框已經是絕對座標
            predBoxes = pred[:, :4]
            predConf = pred[:, 4]
            predCls = pred[:, 5].long()

            # 初始化正確預測陣列
            correct = torch.zeros(len(pred), numIouThresholds, device=device, dtype=torch.bool)

            if numLabels > 0:
                # 計算 IoU
                iou = boxIou(targetBoxes, predBoxes)

                # 對於每個 IoU 閾值
                for iouIdx, iouThresh in enumerate(iouThresholds):
                    # 找到 IoU 超過閾值的配對
                    matches = (iou >= iouThresh) & (targetCls[:, None] == predCls)

                    if matches.any():
                        # 使用貪婪匹配
                        x = torch.where(matches)
                        matchPairs = torch.cat([
                            x[0][:, None],  # 標籤索引
                            x[1][:, None],  # 預測索引
                            iou[x[0], x[1]][:, None]  # IoU
                        ], dim=1)

                        if len(matchPairs) > 1:
                            # 按 IoU 排序
                            matchPairs = matchPairs[matchPairs[:, 2].argsort(descending=True)]

                            # 移除重複的匹配
                            usedLabels = set()
                            usedPreds = set()
                            uniqueMatches = []
                            for m in matchPairs:
                                labelIdx = int(m[0])
                                predIdx = int(m[1])
                                if labelIdx not in usedLabels and predIdx not in usedPreds:
                                    uniqueMatches.append(predIdx)
                                    usedLabels.add(labelIdx)
                                    usedPreds.add(predIdx)
                            correct[uniqueMatches, iouIdx] = True
                        else:
                            correct[int(matchPairs[0, 1]), iouIdx] = True

            # 儲存統計資料
            stats.append((correct.cpu(), predConf.cpu(), predCls.cpu(), targetCls.cpu()))

            # 更新混淆矩陣
            confusionMatrix.processOneBatch(
                pred.cpu().numpy() if pred is not None else None,
                np.column_stack([
                    targetCls.cpu().numpy(),
                    targetBoxes.cpu().numpy()
                ]) if numLabels > 0 else np.zeros((0, 5))
            )

            # 儲存文字結果
            if saveTxt and pred is not None:
                txtPath = labelsDir / (Path(imagePath).stem + '.txt')
                with open(txtPath, 'w') as f:
                    for det in pred.cpu().numpy():
                        classId = int(det[5])
                        conf = det[4]
                        x1, y1, x2, y2 = det[:4]
                        f.write(f'{classId} {conf:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n')

            # 儲存 JSON 結果
            if saveJson and pred is not None:
                # 假設影像 ID 是檔名
                imageId = Path(imagePath).stem
                for det in pred.cpu().numpy():
                    jsonPredictions.append({
                        'image_id': imageId,
                        'category_id': int(det[5]),
                        'bbox': [float(det[0]), float(det[1]),
                                float(det[2] - det[0]), float(det[3] - det[1])],
                        'score': float(det[4])
                    })

    # ============= 計算最終指標 =============
    if len(stats) == 0:
        print('警告: 沒有有效的驗證樣本')
        return {'precision': 0.0, 'recall': 0.0, 'mAP50': 0.0, 'mAP50_95': 0.0}

    # 合併所有統計資料
    stats = [torch.cat(x, 0).cpu().numpy() if isinstance(x[0], torch.Tensor) else np.concatenate(x, 0) for x in zip(*stats)]

    if len(stats) and stats[0].any():
        tp, conf, predCls, targetCls = stats

        # 計算每個類別的 AP
        precision, recall, ap, f1, uniqueCls = apPerClass(
            [tp], [conf], [predCls], [targetCls], numClasses
        )

        # mAP@0.5
        ap50 = ap[:, 0] if len(ap.shape) > 1 else ap
        # mAP@0.5:0.95
        ap50_95 = ap.mean(axis=1) if len(ap.shape) > 1 else ap

        meanPrecision = precision.mean() if len(precision) > 0 else 0.0
        meanRecall = recall.mean() if len(recall) > 0 else 0.0
        mAP50 = ap50.mean() if len(ap50) > 0 else 0.0
        mAP50_95 = ap50_95.mean() if len(ap50_95) > 0 else 0.0
        meanF1 = f1.mean() if len(f1) > 0 else 0.0
    else:
        meanPrecision = meanRecall = mAP50 = mAP50_95 = meanF1 = 0.0
        uniqueCls = []
        ap50 = []

    # 輸出結果
    print('\n' + '='*60)
    print('驗證結果')
    print('='*60)
    print(f'  影像數量: {seen}')
    print(f'  Precision: {meanPrecision:.4f}')
    print(f'  Recall: {meanRecall:.4f}')
    print(f'  F1 Score: {meanF1:.4f}')
    print(f'  mAP@0.5: {mAP50:.4f}')
    print(f'  mAP@0.5:0.95: {mAP50_95:.4f}')

    # 顯示每個類別的結果
    if verbose and len(uniqueCls) > 0:
        print('\n每個類別的 AP@0.5:')
        for i, cls in enumerate(uniqueCls):
            className = classNames[cls] if cls < len(classNames) else f'class_{cls}'
            print(f'  {className}: {ap50[i]:.4f}')

    # 儲存 JSON
    if saveJson and jsonPredictions:
        jsonPath = Path(outputDir) / 'predictions.json'
        with open(jsonPath, 'w') as f:
            json.dump(jsonPredictions, f)
        print(f'\nJSON 結果已儲存到: {jsonPath}')

    # 計算適應度分數
    fitness = fitnessScore({
        'mAP50': mAP50,
        'mAP50_95': mAP50_95
    })
    print(f'\n適應度分數: {fitness:.4f}')
    print('='*60)

    return {
        'precision': meanPrecision,
        'recall': meanRecall,
        'f1': meanF1,
        'mAP50': mAP50,
        'mAP50_95': mAP50_95,
        'fitness': fitness
    }


def xywh2xyxyTorch(xywh: torch.Tensor) -> torch.Tensor:
    """將 xywh 轉換為 xyxy (PyTorch)"""
    xyxy = xywh.clone()
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # x1
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # y1
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # x2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # y2
    return xyxy


def boxIou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """計算 IoU (PyTorch)"""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    interX1 = torch.max(box1[:, None, 0], box2[:, 0])
    interY1 = torch.max(box1[:, None, 1], box2[:, 1])
    interX2 = torch.min(box1[:, None, 2], box2[:, 2])
    interY2 = torch.min(box1[:, None, 3], box2[:, 3])

    interArea = (interX2 - interX1).clamp(0) * (interY2 - interY1).clamp(0)
    iou = interArea / (area1[:, None] + area2 - interArea + 1e-7)

    return iou


def main():
    """主函數"""
    args = parseArgs()

    # 選擇裝置
    device = selectDevice(args.device)

    # 建立輸出目錄
    makeDirectory(args.output)

    # ============= 載入設定 =============
    print('載入設定...')

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
    valPath = dataConfig.get('val')

    if not valPath:
        print('錯誤: 資料設定檔中沒有指定驗證集路徑')
        sys.exit(1)

    print(f'類別數量: {numClasses}')
    print(f'驗證集路徑: {valPath}')

    # ============= 建立資料載入器 =============
    print('\n建立資料載入器...')

    try:
        valLoader, valDataset = createDataLoader(
            path=valPath,
            imageSize=args.imgSize,
            batchSize=args.batchSize,
            augment=False,
            shuffle=False,
            numWorkers=args.workers,
            prefix='驗證: '
        )
    except Exception as e:
        print(f'錯誤: 建立驗證資料載入器時發生錯誤: {e}')
        sys.exit(1)

    # ============= 載入模型 =============
    print('\n載入模型...')

    model = YOLOv7(
        numClasses=numClasses,
        modelType=args.model
    )

    try:
        model = loadWeights(model, args.weights, device=device, strict=False)
    except FileNotFoundError:
        print(f'錯誤: 找不到權重檔案 {args.weights}')
        sys.exit(1)
    except Exception as e:
        print(f'錯誤: 載入權重時發生錯誤: {e}')
        sys.exit(1)

    model = model.to(device)
    model.eval()

    print('模型載入完成')

    # ============= 執行驗證 =============
    metrics = runValidation(
        model=model,
        dataLoader=valLoader,
        device=device,
        numClasses=numClasses,
        classNames=classNames,
        confThreshold=args.confThreshold,
        iouThreshold=args.iouThreshold,
        saveTxt=args.saveTxt,
        saveJson=args.saveJson,
        outputDir=args.output,
        verbose=args.verbose
    )

    # 儲存結果摘要
    summaryPath = Path(args.output) / 'summary.txt'
    with open(summaryPath, 'w') as f:
        f.write('YOLOv7 驗證結果\n')
        f.write('='*40 + '\n')
        f.write(f'權重檔案: {args.weights}\n')
        f.write(f'資料設定: {args.data}\n')
        f.write(f'影像大小: {args.imgSize}\n')
        f.write(f'信心閾值: {args.confThreshold}\n')
        f.write(f'IoU 閾值: {args.iouThreshold}\n')
        f.write('='*40 + '\n')
        for key, value in metrics.items():
            f.write(f'{key}: {value:.4f}\n')

    print(f'\n結果摘要已儲存到: {summaryPath}')


if __name__ == '__main__':
    main()
