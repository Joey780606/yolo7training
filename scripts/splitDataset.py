"""
資料集分割腳本 (Dataset Splitting Script)

此腳本將資料集分割為訓練集、驗證集和測試集。

功能:
1. 按照指定比例隨機分割資料
2. 保持影像和標籤的對應關係
3. 支援分層抽樣（保持類別比例）
4. 可以建立符號連結或複製檔案

使用方式:
    python scripts/splitDataset.py --source ./data --train 0.8 --val 0.2

    # 分割為訓練/驗證/測試三部分
    python scripts/splitDataset.py --source ./data --train 0.7 --val 0.2 --test 0.1
"""

import os
import sys
import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

from tqdm import tqdm


def scanDataset(dataDir: str) -> Tuple[List[str], List[str]]:
    """
    掃描資料集，取得所有影像和標籤檔案

    參數:
        dataDir: 資料集目錄（應包含 images/ 和 labels/ 子目錄）

    返回:
        (影像檔案列表, 標籤檔案列表)
    """
    dataDir = Path(dataDir)

    # 尋找影像目錄
    imagesDir = dataDir / 'images'
    if not imagesDir.exists():
        imagesDir = dataDir

    # 尋找標籤目錄
    labelsDir = dataDir / 'labels'

    # 支援的影像格式
    imageExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']

    # 掃描影像
    imageFiles = []
    for ext in imageExtensions:
        imageFiles.extend(list(imagesDir.glob(f'*{ext}')))
        imageFiles.extend(list(imagesDir.glob(f'*{ext.upper()}')))

    imageFiles = sorted([str(f) for f in imageFiles])

    # 取得對應的標籤檔案
    labelFiles = []
    for imgPath in imageFiles:
        imgPath = Path(imgPath)
        labelPath = labelsDir / (imgPath.stem + '.txt')
        if labelPath.exists():
            labelFiles.append(str(labelPath))
        else:
            labelFiles.append(None)

    return imageFiles, labelFiles


def getClassDistribution(labelFiles: List[str]) -> Dict[int, int]:
    """
    統計類別分布

    參數:
        labelFiles: 標籤檔案列表

    返回:
        類別 ID 到出現次數的字典
    """
    classCount = defaultdict(int)

    for labelFile in labelFiles:
        if labelFile is None:
            continue

        try:
            with open(labelFile, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        classId = int(parts[0])
                        classCount[classId] += 1
        except Exception:
            pass

    return dict(classCount)


def stratifiedSplit(
    imageFiles: List[str],
    labelFiles: List[str],
    trainRatio: float,
    valRatio: float,
    testRatio: float = 0.0,
    seed: int = 42
) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    """
    分層抽樣分割資料集

    根據每張圖片的主要類別進行分層抽樣，
    確保每個分割中的類別比例與原始資料集相似。

    參數:
        imageFiles: 影像檔案列表
        labelFiles: 標籤檔案列表
        trainRatio: 訓練集比例
        valRatio: 驗證集比例
        testRatio: 測試集比例
        seed: 隨機種子

    返回:
        (訓練集列表, 驗證集列表, 測試集列表)
        每個列表的元素為 (影像路徑, 標籤路徑)
    """
    random.seed(seed)

    # 按主要類別分組
    classBuckets = defaultdict(list)

    for imgPath, labelPath in zip(imageFiles, labelFiles):
        # 確定這張圖片的「主要」類別（第一個物件的類別）
        mainClass = -1  # 無標籤的歸為 -1

        if labelPath is not None:
            try:
                with open(labelPath, 'r') as f:
                    firstLine = f.readline().strip()
                    if firstLine:
                        mainClass = int(firstLine.split()[0])
            except Exception:
                pass

        classBuckets[mainClass].append((imgPath, labelPath))

    trainSet = []
    valSet = []
    testSet = []

    # 對每個類別分別進行分割
    for classId, samples in classBuckets.items():
        random.shuffle(samples)
        n = len(samples)

        trainEnd = int(n * trainRatio)
        valEnd = trainEnd + int(n * valRatio)

        trainSet.extend(samples[:trainEnd])
        valSet.extend(samples[trainEnd:valEnd])
        testSet.extend(samples[valEnd:])

    # 再次打亂
    random.shuffle(trainSet)
    random.shuffle(valSet)
    random.shuffle(testSet)

    return trainSet, valSet, testSet


def simpleSplit(
    imageFiles: List[str],
    labelFiles: List[str],
    trainRatio: float,
    valRatio: float,
    testRatio: float = 0.0,
    seed: int = 42
) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    """
    簡單隨機分割資料集

    參數:
        imageFiles: 影像檔案列表
        labelFiles: 標籤檔案列表
        trainRatio: 訓練集比例
        valRatio: 驗證集比例
        testRatio: 測試集比例
        seed: 隨機種子

    返回:
        (訓練集列表, 驗證集列表, 測試集列表)
    """
    random.seed(seed)

    # 配對並打亂
    samples = list(zip(imageFiles, labelFiles))
    random.shuffle(samples)

    n = len(samples)
    trainEnd = int(n * trainRatio)
    valEnd = trainEnd + int(n * valRatio)

    trainSet = samples[:trainEnd]
    valSet = samples[trainEnd:valEnd]
    testSet = samples[valEnd:]

    return trainSet, valSet, testSet


def copyOrLinkFiles(
    samples: List[Tuple[str, str]],
    outputDir: str,
    splitName: str,
    useSymlink: bool = True
) -> int:
    """
    複製或連結檔案到目標目錄

    參數:
        samples: (影像路徑, 標籤路徑) 列表
        outputDir: 輸出目錄
        splitName: 分割名稱 (train/val/test)
        useSymlink: 是否使用符號連結

    返回:
        處理的檔案數量
    """
    outputDir = Path(outputDir)

    # 建立子目錄
    imagesDir = outputDir / 'images' / splitName
    labelsDir = outputDir / 'labels' / splitName
    imagesDir.mkdir(parents=True, exist_ok=True)
    labelsDir.mkdir(parents=True, exist_ok=True)

    count = 0
    for imgPath, labelPath in tqdm(samples, desc=f'處理 {splitName}'):
        imgPath = Path(imgPath)

        # 處理影像
        dstImgPath = imagesDir / imgPath.name

        if not dstImgPath.exists():
            if useSymlink:
                try:
                    os.symlink(imgPath.absolute(), dstImgPath)
                except OSError:
                    shutil.copy2(imgPath, dstImgPath)
            else:
                shutil.copy2(imgPath, dstImgPath)

        # 處理標籤
        if labelPath is not None:
            labelPath = Path(labelPath)
            dstLabelPath = labelsDir / labelPath.name

            if not dstLabelPath.exists():
                if useSymlink:
                    try:
                        os.symlink(labelPath.absolute(), dstLabelPath)
                    except OSError:
                        shutil.copy2(labelPath, dstLabelPath)
                else:
                    shutil.copy2(labelPath, dstLabelPath)

        count += 1

    return count


def createDataYaml(
    outputDir: str,
    classNames: List[str] = None,
    classesFile: str = None
) -> str:
    """
    建立資料集設定檔

    參數:
        outputDir: 輸出目錄
        classNames: 類別名稱列表
        classesFile: 類別名稱檔案路徑

    返回:
        設定檔路徑
    """
    outputDir = Path(outputDir)

    # 讀取類別名稱
    if classNames is None:
        if classesFile and Path(classesFile).exists():
            with open(classesFile, 'r') as f:
                classNames = [line.strip() for line in f if line.strip()]
        else:
            classNames = ['object']  # 預設類別

    # 建立 YAML 內容
    yamlContent = f"""# YOLOv7 資料集設定檔
# 由 splitDataset.py 自動產生

# 資料集路徑
train: {outputDir.absolute()}/images/train
val: {outputDir.absolute()}/images/val
"""

    # 如果有測試集
    testDir = outputDir / 'images' / 'test'
    if testDir.exists() and any(testDir.iterdir()):
        yamlContent += f"test: {outputDir.absolute()}/images/test\n"

    yamlContent += f"""
# 類別數量
nc: {len(classNames)}

# 類別名稱
names:
"""
    for i, name in enumerate(classNames):
        yamlContent += f"  - {name}  # {i}\n"

    # 寫入檔案
    yamlPath = outputDir / 'data.yaml'
    with open(yamlPath, 'w', encoding='utf-8') as f:
        f.write(yamlContent)

    return str(yamlPath)


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='資料集分割工具 - 將資料集分割為訓練/驗證/測試集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
範例:
  # 分割為 80% 訓練、20% 驗證
  python splitDataset.py --source ./data --train 0.8 --val 0.2

  # 分割為 70% 訓練、20% 驗證、10% 測試
  python splitDataset.py --source ./data --train 0.7 --val 0.2 --test 0.1

  # 使用分層抽樣
  python splitDataset.py --source ./data --train 0.8 --val 0.2 --stratified

  # 複製檔案而非建立符號連結
  python splitDataset.py --source ./data --train 0.8 --val 0.2 --copy
        '''
    )

    parser.add_argument('--source', type=str, required=True,
                       help='來源資料目錄（應包含 images/ 和 labels/）')
    parser.add_argument('--output', type=str, default=None,
                       help='輸出目錄（預設為 source 目錄）')
    parser.add_argument('--train', type=float, default=0.8,
                       help='訓練集比例 (預設: 0.8)')
    parser.add_argument('--val', type=float, default=0.2,
                       help='驗證集比例 (預設: 0.2)')
    parser.add_argument('--test', type=float, default=0.0,
                       help='測試集比例 (預設: 0.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='隨機種子 (預設: 42)')
    parser.add_argument('--stratified', action='store_true',
                       help='使用分層抽樣（保持類別比例）')
    parser.add_argument('--copy', action='store_true',
                       help='複製檔案而非建立符號連結')
    parser.add_argument('--classes', type=str, default=None,
                       help='類別名稱檔案')

    args = parser.parse_args()

    # 驗證比例
    totalRatio = args.train + args.val + args.test
    if abs(totalRatio - 1.0) > 0.001:
        print(f'錯誤: 比例總和必須為 1.0，目前為 {totalRatio}')
        sys.exit(1)

    # 設定輸出目錄
    if args.output is None:
        args.output = args.source

    # 掃描資料集
    print('掃描資料集...')
    imageFiles, labelFiles = scanDataset(args.source)
    print(f'找到 {len(imageFiles)} 張影像')

    if len(imageFiles) == 0:
        print('錯誤: 找不到任何影像')
        sys.exit(1)

    # 統計類別分布
    classDist = getClassDistribution(labelFiles)
    print(f'類別分布:')
    for classId, count in sorted(classDist.items()):
        print(f'  類別 {classId}: {count} 個物件')

    # 分割資料集
    print('\n分割資料集...')
    if args.stratified:
        trainSet, valSet, testSet = stratifiedSplit(
            imageFiles, labelFiles, args.train, args.val, args.test, args.seed
        )
    else:
        trainSet, valSet, testSet = simpleSplit(
            imageFiles, labelFiles, args.train, args.val, args.test, args.seed
        )

    print(f'訓練集: {len(trainSet)} 張影像')
    print(f'驗證集: {len(valSet)} 張影像')
    if args.test > 0:
        print(f'測試集: {len(testSet)} 張影像')

    # 複製或連結檔案
    print('\n建立分割目錄...')
    useSymlink = not args.copy

    copyOrLinkFiles(trainSet, args.output, 'train', useSymlink)
    copyOrLinkFiles(valSet, args.output, 'val', useSymlink)
    if args.test > 0:
        copyOrLinkFiles(testSet, args.output, 'test', useSymlink)

    # 建立資料集設定檔
    classesFile = Path(args.source) / 'classes.txt'
    if not classesFile.exists():
        classesFile = args.classes

    yamlPath = createDataYaml(args.output, classesFile=str(classesFile) if classesFile else None)

    print(f'\n分割完成!')
    print(f'  - 輸出目錄: {args.output}')
    print(f'  - 設定檔: {yamlPath}')
    print(f'\n使用方式:')
    print(f'  python train.py --data {yamlPath}')


if __name__ == '__main__':
    main()
