"""
YOLOv7 推論/檢測腳本 (Detection Script)

此腳本用於在圖片、影片或攝影機上執行物件檢測。

功能:
- 支援單張圖片、圖片目錄、影片檔案和即時攝影機
- 輸出帶有邊界框的視覺化結果
- 可選擇儲存檢測結果為文字檔案
- 支援信心閾值和 NMS 閾值調整

使用方式:
    # 檢測單張圖片
    python detect.py --weights runs/train/exp/weights/best.pt --source image.jpg

    # 檢測目錄中的所有圖片
    python detect.py --weights runs/train/exp/weights/best.pt --source ./images/

    # 檢測影片
    python detect.py --weights runs/train/exp/weights/best.pt --source video.mp4

    # 使用攝影機
    python detect.py --weights runs/train/exp/weights/best.pt --source 0
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch

# 添加專案根目錄到路徑
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.yolo import YOLOv7, loadWeights
from utils.general import (
    nonMaxSuppression, scaleBoxes, colorList,
    makeDirectory
)
from utils.torchUtils import selectDevice
from utils.augmentations import letterbox


def parseArgs():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='YOLOv7 物件檢測程式',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 基本設定
    parser.add_argument('--weights', type=str, required=True,
                       help='模型權重路徑')
    parser.add_argument('--source', type=str, required=True,
                       help='輸入來源 (圖片路徑、目錄、影片或攝影機編號)')
    parser.add_argument('--output', type=str, default='runs/detect',
                       help='輸出目錄')

    # 模型設定
    parser.add_argument('--imgSize', type=int, default=640,
                       help='推論影像大小')
    parser.add_argument('--model', type=str, default='yolov7',
                       choices=['yolov7', 'yolov7tiny'],
                       help='模型類型')
    parser.add_argument('--numClasses', type=int, default=80,
                       help='類別數量')

    # 推論設定
    parser.add_argument('--confThreshold', type=float, default=0.25,
                       help='信心閾值')
    parser.add_argument('--iouThreshold', type=float, default=0.45,
                       help='NMS IoU 閾值')
    parser.add_argument('--device', type=str, default='',
                       help='計算裝置')

    # 輸出設定
    parser.add_argument('--saveTxt', action='store_true',
                       help='儲存檢測結果為文字檔')
    parser.add_argument('--saveImg', action='store_true', default=True,
                       help='儲存帶有邊界框的影像')
    parser.add_argument('--view', action='store_true',
                       help='即時顯示結果')
    parser.add_argument('--hideLabels', action='store_true',
                       help='隱藏標籤文字')
    parser.add_argument('--hideConf', action='store_true',
                       help='隱藏信心分數')

    # 類別設定
    parser.add_argument('--classes', type=str, default=None,
                       help='類別名稱檔案路徑')

    return parser.parse_args()


def loadClassNames(classFile: Optional[str], numClasses: int) -> List[str]:
    """
    載入類別名稱

    參數:
        classFile: 類別名稱檔案路徑
        numClasses: 類別數量

    返回:
        類別名稱列表
    """
    if classFile and os.path.isfile(classFile):
        try:
            with open(classFile, 'r', encoding='utf-8') as f:
                names = [line.strip() for line in f if line.strip()]
            return names
        except Exception:
            pass

    # 使用預設的 COCO 類別名稱
    cocoNames = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    if numClasses <= len(cocoNames):
        return cocoNames[:numClasses]
    else:
        return [f'class_{i}' for i in range(numClasses)]


def preprocessImage(
    image: np.ndarray,
    imgSize: int,
    device: torch.device
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[float, float], Tuple[float, float]]:
    """
    預處理影像

    參數:
        image: 輸入影像 (BGR)
        imgSize: 目標大小
        device: 計算裝置

    返回:
        (處理後的張量, 原始尺寸, 縮放比例, 填充量)
    """
    originalShape = image.shape[:2]  # [height, width]

    # Letterbox 縮放
    imgLetterbox, ratio, pad = letterbox(image, (imgSize, imgSize), auto=False)

    # BGR -> RGB
    imgRgb = imgLetterbox[:, :, ::-1]

    # HWC -> CHW
    imgChw = imgRgb.transpose((2, 0, 1))

    # 轉換為張量並正規化
    imgTensor = torch.from_numpy(np.ascontiguousarray(imgChw)).float()
    imgTensor /= 255.0

    # 添加批次維度
    if imgTensor.ndimension() == 3:
        imgTensor = imgTensor.unsqueeze(0)

    return imgTensor.to(device), originalShape, ratio, pad


def drawDetections(
    image: np.ndarray,
    detections: np.ndarray,
    classNames: List[str],
    colors: List[Tuple[int, int, int]],
    hideLabels: bool = False,
    hideConf: bool = False
) -> np.ndarray:
    """
    在影像上繪製檢測結果

    參數:
        image: 原始影像
        detections: 檢測結果 [N, 6] (x1, y1, x2, y2, conf, class)
        classNames: 類別名稱列表
        colors: 顏色列表
        hideLabels: 是否隱藏標籤
        hideConf: 是否隱藏信心分數

    返回:
        繪製後的影像
    """
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        conf = det[4]
        classId = int(det[5])

        # 取得顏色
        color = colors[classId % len(colors)]

        # 繪製邊界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 繪製標籤
        if not hideLabels:
            className = classNames[classId] if classId < len(classNames) else f'class_{classId}'
            if hideConf:
                label = className
            else:
                label = f'{className} {conf:.2f}'

            # 計算標籤大小
            (labelWidth, labelHeight), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # 繪製標籤背景
            cv2.rectangle(
                image,
                (x1, y1 - labelHeight - 10),
                (x1 + labelWidth, y1),
                color, -1
            )

            # 繪製標籤文字
            cv2.putText(
                image, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1
            )

    return image


def detectImage(
    model: torch.nn.Module,
    imagePath: str,
    imgSize: int,
    device: torch.device,
    confThreshold: float,
    iouThreshold: float,
    classNames: List[str],
    colors: List[Tuple[int, int, int]],
    outputDir: str,
    saveTxt: bool,
    saveImg: bool,
    hideLabels: bool,
    hideConf: bool
) -> Tuple[int, float]:
    """
    檢測單張影像

    參數:
        model: 模型
        imagePath: 影像路徑
        imgSize: 推論影像大小
        device: 計算裝置
        confThreshold: 信心閾值
        iouThreshold: NMS IoU 閾值
        classNames: 類別名稱
        colors: 顏色列表
        outputDir: 輸出目錄
        saveTxt: 是否儲存文字
        saveImg: 是否儲存影像
        hideLabels: 隱藏標籤
        hideConf: 隱藏信心

    返回:
        (檢測到的物體數量, 推論時間)
    """
    # 讀取影像
    image = cv2.imread(imagePath)
    if image is None:
        print(f'警告: 無法讀取影像 {imagePath}')
        return 0, 0.0

    # 預處理
    imgTensor, originalShape, ratio, pad = preprocessImage(image, imgSize, device)

    # 推論
    startTime = time.time()
    with torch.no_grad():
        predictions, _ = model(imgTensor)
    inferenceTime = time.time() - startTime

    # NMS
    detections = nonMaxSuppression(
        predictions,
        confThreshold=confThreshold,
        iouThreshold=iouThreshold
    )[0]

    numDetections = 0

    if detections is not None and len(detections) > 0:
        # 將座標轉換回原始圖片尺寸
        detections[:, :4] = scaleBoxes(
            (imgSize, imgSize),
            detections[:, :4],
            originalShape
        ).round()

        numDetections = len(detections)

        # 轉換為 numpy
        detections = detections.cpu().numpy()

        # 繪製結果
        resultImage = drawDetections(
            image.copy(), detections, classNames, colors, hideLabels, hideConf
        )

        # 儲存結果
        fileName = Path(imagePath).name

        if saveImg:
            outputPath = os.path.join(outputDir, fileName)
            cv2.imwrite(outputPath, resultImage)

        if saveTxt:
            txtPath = os.path.join(outputDir, Path(fileName).stem + '.txt')
            with open(txtPath, 'w') as f:
                for det in detections:
                    classId = int(det[5])
                    conf = det[4]
                    x1, y1, x2, y2 = det[:4]
                    # 轉換為 YOLO 格式（正規化的 xywh）
                    h, w = originalShape
                    xCenter = (x1 + x2) / 2 / w
                    yCenter = (y1 + y2) / 2 / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    f.write(f'{classId} {xCenter:.6f} {yCenter:.6f} {width:.6f} {height:.6f} {conf:.4f}\n')

    return numDetections, inferenceTime


def detectVideo(
    model: torch.nn.Module,
    videoPath: str,
    imgSize: int,
    device: torch.device,
    confThreshold: float,
    iouThreshold: float,
    classNames: List[str],
    colors: List[Tuple[int, int, int]],
    outputDir: str,
    saveVideo: bool,
    viewVideo: bool,
    hideLabels: bool,
    hideConf: bool
) -> None:
    """
    檢測影片

    參數:
        model: 模型
        videoPath: 影片路徑或攝影機編號
        其他參數同 detectImage
    """
    # 開啟影片來源
    if videoPath.isdigit():
        cap = cv2.VideoCapture(int(videoPath))
        isCamera = True
    else:
        cap = cv2.VideoCapture(videoPath)
        isCamera = False

    if not cap.isOpened():
        print(f'錯誤: 無法開啟影片來源 {videoPath}')
        return

    # 取得影片資訊
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # 設定影片輸出
    writer = None
    if saveVideo and not isCamera:
        outputPath = os.path.join(outputDir, Path(videoPath).stem + '_detected.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(outputPath, fourcc, fps, (frameWidth, frameHeight))

    frameCount = 0
    totalTime = 0

    print('開始處理影片... (按 q 退出)')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frameCount += 1

        # 預處理
        imgTensor, originalShape, ratio, pad = preprocessImage(frame, imgSize, device)

        # 推論
        startTime = time.time()
        with torch.no_grad():
            predictions, _ = model(imgTensor)
        inferenceTime = time.time() - startTime
        totalTime += inferenceTime

        # NMS
        detections = nonMaxSuppression(
            predictions,
            confThreshold=confThreshold,
            iouThreshold=iouThreshold
        )[0]

        # 處理檢測結果
        if detections is not None and len(detections) > 0:
            detections[:, :4] = scaleBoxes(
                (imgSize, imgSize),
                detections[:, :4],
                originalShape
            ).round()
            detections = detections.cpu().numpy()

            frame = drawDetections(
                frame, detections, classNames, colors, hideLabels, hideConf
            )

        # 顯示 FPS
        currentFps = 1.0 / inferenceTime if inferenceTime > 0 else 0
        cv2.putText(
            frame, f'FPS: {currentFps:.1f}',
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        # 儲存影片
        if writer is not None:
            writer.write(frame)

        # 顯示結果
        if viewVideo:
            cv2.imshow('YOLOv7 Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 清理
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # 輸出統計
    avgFps = frameCount / totalTime if totalTime > 0 else 0
    print(f'\n處理完成!')
    print(f'  總幀數: {frameCount}')
    print(f'  平均 FPS: {avgFps:.1f}')


def main():
    """主函數"""
    args = parseArgs()

    # 選擇裝置
    device = selectDevice(args.device)

    # 載入類別名稱
    classNames = loadClassNames(args.classes, args.numClasses)

    # 產生顏色
    colors = colorList()

    # 建立輸出目錄
    makeDirectory(args.output)

    # ============= 載入模型 =============
    print('載入模型...')

    model = YOLOv7(
        numClasses=args.numClasses,
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

    print(f'模型載入完成')

    # ============= 判斷輸入類型 =============
    source = args.source

    # 檢查是否為攝影機
    if source.isdigit():
        print('使用攝影機進行即時檢測...')
        detectVideo(
            model, source, args.imgSize, device,
            args.confThreshold, args.iouThreshold,
            classNames, colors, args.output,
            saveVideo=False, viewVideo=True,
            hideLabels=args.hideLabels, hideConf=args.hideConf
        )
        return

    sourcePath = Path(source)

    # 檢查是否為影片
    videoExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    if sourcePath.suffix.lower() in videoExtensions:
        print(f'處理影片: {source}')
        detectVideo(
            model, source, args.imgSize, device,
            args.confThreshold, args.iouThreshold,
            classNames, colors, args.output,
            saveVideo=args.saveImg, viewVideo=args.view,
            hideLabels=args.hideLabels, hideConf=args.hideConf
        )
        return

    # 處理圖片
    imageExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']

    if sourcePath.is_file():
        imageFiles = [source]
    elif sourcePath.is_dir():
        imageFiles = []
        for ext in imageExtensions:
            imageFiles.extend(list(sourcePath.glob(f'*{ext}')))
            imageFiles.extend(list(sourcePath.glob(f'*{ext.upper()}')))
        imageFiles = sorted([str(f) for f in imageFiles])
    else:
        print(f'錯誤: 無效的輸入來源 {source}')
        sys.exit(1)

    if len(imageFiles) == 0:
        print(f'錯誤: 在 {source} 找不到圖片')
        sys.exit(1)

    print(f'找到 {len(imageFiles)} 張圖片')

    # 處理每張圖片
    totalDetections = 0
    totalTime = 0

    for imgPath in imageFiles:
        numDets, infTime = detectImage(
            model, imgPath, args.imgSize, device,
            args.confThreshold, args.iouThreshold,
            classNames, colors, args.output,
            args.saveTxt, args.saveImg,
            args.hideLabels, args.hideConf
        )
        totalDetections += numDets
        totalTime += infTime

        print(f'  {Path(imgPath).name}: {numDets} 個物體, {infTime*1000:.1f} ms')

    # 輸出統計
    avgTime = totalTime / len(imageFiles) * 1000 if imageFiles else 0
    print(f'\n完成!')
    print(f'  總檢測數: {totalDetections}')
    print(f'  平均推論時間: {avgTime:.1f} ms/張')
    print(f'  結果儲存在: {args.output}')


if __name__ == '__main__':
    main()
