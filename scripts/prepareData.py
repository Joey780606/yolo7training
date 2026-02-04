"""
資料準備腳本 (Data Preparation Script)

此腳本用於將常見的標註格式轉換為 YOLO 格式。

支援的輸入格式:
1. VOC (Pascal VOC XML 格式)
2. COCO (JSON 格式)
3. 自訂 CSV 格式

YOLO 標籤格式說明:
- 每張圖片對應一個 .txt 標籤檔案
- 每行代表一個物件: <class_id> <x_center> <y_center> <width> <height>
- 所有座標都是相對於圖片尺寸的正規化值 (0.0 ~ 1.0)

使用方式:
    python scripts/prepareData.py --source ./rawData --format voc --output ./data
"""

import os
import sys
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import cv2
from tqdm import tqdm


def convertVocToYolo(
    vocDir: str,
    outputDir: str,
    classNames: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    將 Pascal VOC 格式轉換為 YOLO 格式

    VOC 格式說明:
    - 影像通常在 JPEGImages/ 目錄
    - 標註在 Annotations/ 目錄，為 XML 檔案
    - XML 中包含 <object> 元素，每個代表一個物件
    - 邊界框使用 <bndbox> 元素，包含 xmin, ymin, xmax, ymax

    參數:
        vocDir: VOC 資料集目錄
        outputDir: 輸出目錄
        classNames: 類別名稱列表（如果為 None，將自動建立）

    返回:
        類別名稱到 ID 的映射字典
    """
    vocDir = Path(vocDir)
    outputDir = Path(outputDir)

    # 尋找標註目錄
    annotationsDir = vocDir / 'Annotations'
    if not annotationsDir.exists():
        annotationsDir = vocDir  # 可能標註直接在根目錄

    # 尋找影像目錄
    imagesDir = vocDir / 'JPEGImages'
    if not imagesDir.exists():
        imagesDir = vocDir / 'images'
    if not imagesDir.exists():
        imagesDir = vocDir

    # 建立輸出目錄
    outputImagesDir = outputDir / 'images'
    outputLabelsDir = outputDir / 'labels'
    outputImagesDir.mkdir(parents=True, exist_ok=True)
    outputLabelsDir.mkdir(parents=True, exist_ok=True)

    # 收集所有類別名稱
    if classNames is None:
        print('掃描類別名稱...')
        classNamesSet = set()
        for xmlFile in annotationsDir.glob('*.xml'):
            try:
                tree = ET.parse(xmlFile)
                root = tree.getroot()
                for obj in root.findall('object'):
                    className = obj.find('name').text
                    classNamesSet.add(className)
            except Exception as e:
                print(f'警告: 解析 {xmlFile} 時發生錯誤: {e}')

        classNames = sorted(list(classNamesSet))
        print(f'找到 {len(classNames)} 個類別: {classNames}')

    # 建立類別映射
    classToId = {name: i for i, name in enumerate(classNames)}

    # 轉換標註
    print('轉換標註中...')
    xmlFiles = list(annotationsDir.glob('*.xml'))

    for xmlFile in tqdm(xmlFiles, desc='轉換 VOC 標註'):
        try:
            tree = ET.parse(xmlFile)
            root = tree.getroot()

            # 取得影像檔名和尺寸
            filename = root.find('filename').text

            # 嘗試取得尺寸，如果 XML 中沒有則讀取影像
            sizeElement = root.find('size')
            if sizeElement is not None:
                imgWidth = int(sizeElement.find('width').text)
                imgHeight = int(sizeElement.find('height').text)
            else:
                # 從影像讀取尺寸
                imgPath = imagesDir / filename
                if imgPath.exists():
                    img = cv2.imread(str(imgPath))
                    if img is not None:
                        imgHeight, imgWidth = img.shape[:2]
                    else:
                        print(f'警告: 無法讀取影像 {imgPath}')
                        continue
                else:
                    print(f'警告: 找不到影像 {imgPath}')
                    continue

            # 轉換每個物件
            yoloLabels = []
            for obj in root.findall('object'):
                className = obj.find('name').text

                # 跳過 difficult 物件（可選）
                difficult = obj.find('difficult')
                if difficult is not None and int(difficult.text) == 1:
                    continue

                if className not in classToId:
                    print(f'警告: 未知類別 {className}')
                    continue

                classId = classToId[className]

                # 取得邊界框
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                # 轉換為 YOLO 格式 (正規化的中心點和寬高)
                xCenter = (xmin + xmax) / 2 / imgWidth
                yCenter = (ymin + ymax) / 2 / imgHeight
                width = (xmax - xmin) / imgWidth
                height = (ymax - ymin) / imgHeight

                # 確保座標在有效範圍內
                xCenter = max(0, min(1, xCenter))
                yCenter = max(0, min(1, yCenter))
                width = max(0, min(1, width))
                height = max(0, min(1, height))

                yoloLabels.append(f'{classId} {xCenter:.6f} {yCenter:.6f} {width:.6f} {height:.6f}')

            # 寫入標籤檔案
            labelFilename = xmlFile.stem + '.txt'
            labelPath = outputLabelsDir / labelFilename

            with open(labelPath, 'w') as f:
                f.write('\n'.join(yoloLabels))

            # 複製或連結影像
            srcImgPath = imagesDir / filename
            dstImgPath = outputImagesDir / filename

            if srcImgPath.exists() and not dstImgPath.exists():
                # 建立符號連結或複製
                try:
                    os.symlink(srcImgPath.absolute(), dstImgPath)
                except OSError:
                    import shutil
                    shutil.copy2(srcImgPath, dstImgPath)

        except Exception as e:
            print(f'錯誤: 處理 {xmlFile} 時發生錯誤: {e}')

    # 儲存類別名稱
    classesPath = outputDir / 'classes.txt'
    with open(classesPath, 'w') as f:
        f.write('\n'.join(classNames))

    print(f'\n轉換完成!')
    print(f'  - 影像目錄: {outputImagesDir}')
    print(f'  - 標籤目錄: {outputLabelsDir}')
    print(f'  - 類別列表: {classesPath}')

    return classToId


def convertCocoToYolo(
    cocoJsonPath: str,
    imagesDir: str,
    outputDir: str
) -> Dict[str, int]:
    """
    將 COCO 格式轉換為 YOLO 格式

    COCO 格式說明:
    - 所有標註都在一個 JSON 檔案中
    - 包含 images, annotations, categories 三個主要欄位
    - annotations 中的 bbox 格式為 [x, y, width, height]（左上角座標）

    參數:
        cocoJsonPath: COCO JSON 檔案路徑
        imagesDir: 影像目錄
        outputDir: 輸出目錄

    返回:
        類別名稱到 ID 的映射字典
    """
    outputDir = Path(outputDir)
    imagesDir = Path(imagesDir)

    # 建立輸出目錄
    outputImagesDir = outputDir / 'images'
    outputLabelsDir = outputDir / 'labels'
    outputImagesDir.mkdir(parents=True, exist_ok=True)
    outputLabelsDir.mkdir(parents=True, exist_ok=True)

    # 載入 COCO JSON
    print(f'載入 COCO 標註: {cocoJsonPath}')
    try:
        with open(cocoJsonPath, 'r') as f:
            coco = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f'找不到 COCO JSON 檔案: {cocoJsonPath}')
    except json.JSONDecodeError as e:
        raise ValueError(f'COCO JSON 格式錯誤: {e}')

    # 解析類別
    categories = coco.get('categories', [])
    categoryIdToName = {cat['id']: cat['name'] for cat in categories}
    classNames = [cat['name'] for cat in sorted(categories, key=lambda x: x['id'])]
    classToId = {name: i for i, name in enumerate(classNames)}

    # 建立新的類別 ID 映射（COCO 的 category_id 可能不是從 0 開始的）
    oldIdToNewId = {cat['id']: classToId[cat['name']] for cat in categories}

    print(f'類別數量: {len(classNames)}')

    # 建立影像 ID 到影像資訊的映射
    images = coco.get('images', [])
    imageIdToInfo = {img['id']: img for img in images}

    # 按影像分組標註
    annotations = coco.get('annotations', [])
    imageIdToAnnotations = defaultdict(list)
    for ann in annotations:
        imageIdToAnnotations[ann['image_id']].append(ann)

    # 轉換標註
    print('轉換標註中...')
    for imageId, imgInfo in tqdm(imageIdToInfo.items(), desc='轉換 COCO 標註'):
        filename = imgInfo['file_name']
        imgWidth = imgInfo['width']
        imgHeight = imgInfo['height']

        # 轉換該影像的所有標註
        yoloLabels = []
        for ann in imageIdToAnnotations[imageId]:
            # 跳過群組標註
            if ann.get('iscrowd', 0) == 1:
                continue

            categoryId = ann['category_id']
            if categoryId not in oldIdToNewId:
                continue

            newClassId = oldIdToNewId[categoryId]

            # COCO bbox: [x, y, width, height]（左上角）
            bbox = ann['bbox']
            x, y, w, h = bbox

            # 轉換為 YOLO 格式
            xCenter = (x + w / 2) / imgWidth
            yCenter = (y + h / 2) / imgHeight
            width = w / imgWidth
            height = h / imgHeight

            # 確保座標在有效範圍內
            xCenter = max(0, min(1, xCenter))
            yCenter = max(0, min(1, yCenter))
            width = max(0, min(1, width))
            height = max(0, min(1, height))

            yoloLabels.append(f'{newClassId} {xCenter:.6f} {yCenter:.6f} {width:.6f} {height:.6f}')

        # 寫入標籤檔案
        labelFilename = Path(filename).stem + '.txt'
        labelPath = outputLabelsDir / labelFilename

        with open(labelPath, 'w') as f:
            f.write('\n'.join(yoloLabels))

        # 複製或連結影像
        srcImgPath = imagesDir / filename
        dstImgPath = outputImagesDir / filename

        if srcImgPath.exists() and not dstImgPath.exists():
            try:
                os.symlink(srcImgPath.absolute(), dstImgPath)
            except OSError:
                import shutil
                shutil.copy2(srcImgPath, dstImgPath)

    # 儲存類別名稱
    classesPath = outputDir / 'classes.txt'
    with open(classesPath, 'w') as f:
        f.write('\n'.join(classNames))

    print(f'\n轉換完成!')
    print(f'  - 影像目錄: {outputImagesDir}')
    print(f'  - 標籤目錄: {outputLabelsDir}')
    print(f'  - 類別列表: {classesPath}')

    return classToId


def convertCsvToYolo(
    csvPath: str,
    outputDir: str,
    imageColumn: str = 'image',
    classColumn: str = 'class',
    xminColumn: str = 'xmin',
    yminColumn: str = 'ymin',
    xmaxColumn: str = 'xmax',
    ymaxColumn: str = 'ymax',
    widthColumn: str = 'width',
    heightColumn: str = 'height'
) -> Dict[str, int]:
    """
    將 CSV 格式轉換為 YOLO 格式

    CSV 預期欄位:
    - image: 影像檔案路徑
    - class: 類別名稱
    - xmin, ymin, xmax, ymax: 邊界框座標（絕對值）
    - width, height: 影像尺寸

    參數:
        csvPath: CSV 檔案路徑
        outputDir: 輸出目錄
        各欄位名稱參數

    返回:
        類別名稱到 ID 的映射字典
    """
    import pandas as pd

    outputDir = Path(outputDir)

    # 建立輸出目錄
    outputLabelsDir = outputDir / 'labels'
    outputLabelsDir.mkdir(parents=True, exist_ok=True)

    # 讀取 CSV
    print(f'載入 CSV: {csvPath}')
    try:
        df = pd.read_csv(csvPath)
    except FileNotFoundError:
        raise FileNotFoundError(f'找不到 CSV 檔案: {csvPath}')
    except Exception as e:
        raise RuntimeError(f'讀取 CSV 時發生錯誤: {e}')

    # 取得所有類別
    classNames = sorted(df[classColumn].unique().tolist())
    classToId = {name: i for i, name in enumerate(classNames)}

    print(f'類別數量: {len(classNames)}')

    # 按影像分組
    grouped = df.groupby(imageColumn)

    print('轉換標註中...')
    for imagePath, group in tqdm(grouped, desc='轉換 CSV 標註'):
        yoloLabels = []

        for _, row in group.iterrows():
            className = row[classColumn]
            classId = classToId[className]

            imgWidth = row[widthColumn]
            imgHeight = row[heightColumn]

            xmin = row[xminColumn]
            ymin = row[yminColumn]
            xmax = row[xmaxColumn]
            ymax = row[ymaxColumn]

            # 轉換為 YOLO 格式
            xCenter = (xmin + xmax) / 2 / imgWidth
            yCenter = (ymin + ymax) / 2 / imgHeight
            width = (xmax - xmin) / imgWidth
            height = (ymax - ymin) / imgHeight

            yoloLabels.append(f'{classId} {xCenter:.6f} {yCenter:.6f} {width:.6f} {height:.6f}')

        # 寫入標籤檔案
        labelFilename = Path(imagePath).stem + '.txt'
        labelPath = outputLabelsDir / labelFilename

        with open(labelPath, 'w') as f:
            f.write('\n'.join(yoloLabels))

    # 儲存類別名稱
    classesPath = outputDir / 'classes.txt'
    with open(classesPath, 'w') as f:
        f.write('\n'.join(classNames))

    print(f'\n轉換完成!')
    print(f'  - 標籤目錄: {outputLabelsDir}')
    print(f'  - 類別列表: {classesPath}')

    return classToId


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='資料格式轉換工具 - 將常見標註格式轉換為 YOLO 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
範例:
  # 轉換 VOC 格式
  python prepareData.py --source ./VOCdevkit/VOC2012 --format voc --output ./data

  # 轉換 COCO 格式
  python prepareData.py --source ./annotations.json --images ./images --format coco --output ./data

  # 轉換 CSV 格式
  python prepareData.py --source ./annotations.csv --format csv --output ./data
        '''
    )

    parser.add_argument('--source', type=str, required=True,
                       help='來源資料路徑（VOC 目錄、COCO JSON 或 CSV 檔案）')
    parser.add_argument('--format', type=str, required=True, choices=['voc', 'coco', 'csv'],
                       help='來源資料格式')
    parser.add_argument('--output', type=str, required=True,
                       help='輸出目錄')
    parser.add_argument('--images', type=str, default=None,
                       help='影像目錄（COCO 格式需要）')
    parser.add_argument('--classes', type=str, default=None,
                       help='類別名稱檔案（每行一個類別名稱）')

    args = parser.parse_args()

    # 載入類別名稱（如果提供）
    classNames = None
    if args.classes:
        try:
            with open(args.classes, 'r') as f:
                classNames = [line.strip() for line in f.readlines() if line.strip()]
        except FileNotFoundError:
            print(f'警告: 找不到類別檔案 {args.classes}，將自動偵測類別')

    # 執行轉換
    try:
        if args.format == 'voc':
            convertVocToYolo(args.source, args.output, classNames)

        elif args.format == 'coco':
            if args.images is None:
                # 嘗試從 source 路徑推斷影像目錄
                sourcePath = Path(args.source)
                args.images = str(sourcePath.parent / 'images')
                print(f'未指定 --images，使用預設: {args.images}')

            convertCocoToYolo(args.source, args.images, args.output)

        elif args.format == 'csv':
            convertCsvToYolo(args.source, args.output)

    except Exception as e:
        print(f'錯誤: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
