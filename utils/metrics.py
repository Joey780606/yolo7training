"""
評估指標模組 (Metrics Module)

此模組實現了物件檢測的評估指標，包括：
- Precision (精確率): 預測為正的樣本中，實際為正的比例
- Recall (召回率): 實際為正的樣本中，被正確預測的比例
- AP (Average Precision): 精確率-召回率曲線下的面積
- mAP (mean Average Precision): 所有類別 AP 的平均值
- F1 Score: 精確率和召回率的調和平均

這些指標是評估物件檢測模型性能的標準方法。

常用的 mAP 版本:
- mAP@0.5: IoU 閾值為 0.5 的 mAP
- mAP@0.5:0.95: IoU 閾值從 0.5 到 0.95 的平均 mAP
"""

from typing import List, Tuple, Dict, Optional
import numpy as np


def computeAp(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    計算單一類別的 AP (Average Precision)

    AP 是精確率-召回率曲線 (PR Curve) 下的面積。
    使用 11 點插值法或所有點插值法計算。

    參數:
        recall: 召回率陣列，按升序排列
        precision: 對應的精確率陣列

    返回:
        AP 值
    """
    # 在首尾添加點確保曲線完整
    # recall: [0, ..., 1]
    # precision: [1, ..., 0]
    mrecall = np.concatenate(([0.0], recall, [1.0]))
    mprecision = np.concatenate(([1.0], precision, [0.0]))

    # 確保 precision 是單調遞減的
    # 從右往左，取每個點右邊的最大值
    for i in range(len(mprecision) - 2, -1, -1):
        mprecision[i] = max(mprecision[i], mprecision[i + 1])

    # 找到 recall 變化的點
    indices = np.where(mrecall[1:] != mrecall[:-1])[0]

    # 計算曲線下面積
    ap = np.sum((mrecall[indices + 1] - mrecall[indices]) * mprecision[indices + 1])

    return ap


def computeIou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    計算兩組框之間的 IoU

    參數:
        box1: 第一組框 [N, 4] (x1, y1, x2, y2)
        box2: 第二組框 [M, 4] (x1, y1, x2, y2)

    返回:
        IoU 矩陣 [N, M]
    """
    # 計算面積
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # 計算交集
    interX1 = np.maximum(box1[:, None, 0], box2[:, 0])
    interY1 = np.maximum(box1[:, None, 1], box2[:, 1])
    interX2 = np.minimum(box1[:, None, 2], box2[:, 2])
    interY2 = np.minimum(box1[:, None, 3], box2[:, 3])

    interArea = np.maximum(interX2 - interX1, 0) * np.maximum(interY2 - interY1, 0)

    # 計算 IoU
    iou = interArea / (area1[:, None] + area2 - interArea + 1e-7)

    return iou


class ConfusionMatrix:
    """
    混淆矩陣

    混淆矩陣用於分析模型的分類性能，
    可以看出模型在哪些類別上容易混淆。

    矩陣結構:
    - 行: 實際類別
    - 列: 預測類別
    - 對角線: 正確分類
    - 非對角線: 錯誤分類

    屬性:
        matrix: 混淆矩陣 [numClasses + 1, numClasses + 1]
                最後一行/列用於背景類別
        numClasses: 類別數量
    """

    def __init__(self, numClasses: int, confThreshold: float = 0.25, iouThreshold: float = 0.45):
        """
        初始化混淆矩陣

        參數:
            numClasses: 類別數量
            confThreshold: 信心閾值
            iouThreshold: IoU 閾值
        """
        self.matrix = np.zeros((numClasses + 1, numClasses + 1))
        self.numClasses = numClasses
        self.confThreshold = confThreshold
        self.iouThreshold = iouThreshold

    def processOneBatch(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> None:
        """
        處理一個批次的預測結果

        參數:
            predictions: 預測框 [N, 6] (x1, y1, x2, y2, conf, class)
            labels: 真實標籤 [M, 5] (class, x1, y1, x2, y2)
        """
        if predictions is None or len(predictions) == 0:
            # 沒有預測，所有真實標籤都是漏檢 (FN)
            for label in labels:
                self.matrix[int(label[0]), self.numClasses] += 1
            return

        # 過濾低信心預測
        predictions = predictions[predictions[:, 4] > self.confThreshold]

        # 取得預測類別
        predClasses = predictions[:, 5].astype(int)
        predBoxes = predictions[:, :4]

        # 取得真實類別
        trueClasses = labels[:, 0].astype(int)
        trueBoxes = labels[:, 1:]

        # 計算 IoU
        iou = computeIou(trueBoxes, predBoxes)

        # 為每個真實框找到最佳匹配
        matchedPreds = set()

        for i, (trueClass, trueIou) in enumerate(zip(trueClasses, iou)):
            # 找到 IoU 超過閾值且類別匹配的預測
            matches = np.where((trueIou > self.iouThreshold) & (predClasses == trueClass))[0]

            if len(matches) > 0:
                # 選擇 IoU 最大的匹配
                bestMatch = matches[np.argmax(trueIou[matches])]
                if bestMatch not in matchedPreds:
                    self.matrix[trueClass, trueClass] += 1  # TP
                    matchedPreds.add(bestMatch)
                else:
                    # 已經匹配過，算作 FN
                    self.matrix[trueClass, self.numClasses] += 1
            else:
                # 沒有匹配，可能是類別錯誤或漏檢
                wrongClassMatches = np.where(trueIou > self.iouThreshold)[0]
                if len(wrongClassMatches) > 0:
                    # 類別錯誤
                    wrongPred = wrongClassMatches[np.argmax(trueIou[wrongClassMatches])]
                    self.matrix[trueClass, predClasses[wrongPred]] += 1
                else:
                    # 漏檢
                    self.matrix[trueClass, self.numClasses] += 1

        # 處理誤檢 (FP)
        for j, predClass in enumerate(predClasses):
            if j not in matchedPreds:
                self.matrix[self.numClasses, predClass] += 1

    def getMatrix(self) -> np.ndarray:
        """返回混淆矩陣"""
        return self.matrix

    def printMatrix(self, names: List[str] = None) -> None:
        """
        列印混淆矩陣

        參數:
            names: 類別名稱列表
        """
        if names is None:
            names = [str(i) for i in range(self.numClasses)]
        names = names + ['背景']

        print('\n混淆矩陣:')
        print('  ' + ''.join(f'{n:>10}' for n in names))
        for i, row in enumerate(self.matrix):
            print(f'{names[i]:>10}' + ''.join(f'{int(x):>10}' for x in row))


class MetricsCalculator:
    """
    評估指標計算器

    這個類別負責累積所有預測結果並計算最終指標。

    使用方式:
    1. 初始化計算器
    2. 對每個批次調用 update()
    3. 調用 compute() 取得最終結果

    屬性:
        numClasses: 類別數量
        iouThresholds: IoU 閾值列表
        allPredictions: 累積的所有預測
        allLabels: 累積的所有標籤
    """

    def __init__(
        self,
        numClasses: int,
        iouThresholds: np.ndarray = None
    ):
        """
        初始化指標計算器

        參數:
            numClasses: 類別數量
            iouThresholds: IoU 閾值列表（用於 mAP@0.5:0.95）
        """
        self.numClasses = numClasses

        # 預設使用 COCO 的 10 個 IoU 閾值
        if iouThresholds is None:
            iouThresholds = np.linspace(0.5, 0.95, 10)
        self.iouThresholds = iouThresholds

        # 累積預測和標籤
        self.allPredictions = []  # [(imageId, classId, confidence, x1, y1, x2, y2), ...]
        self.allLabels = []  # [(imageId, classId, x1, y1, x2, y2), ...]

        self.currentImageId = 0

    def update(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> None:
        """
        更新累積的預測和標籤

        參數:
            predictions: 預測框 [N, 6] (x1, y1, x2, y2, conf, class)
            labels: 真實標籤 [M, 5] (class, x, y, w, h) 正規化座標
        """
        # 處理預測
        if predictions is not None and len(predictions) > 0:
            for pred in predictions:
                self.allPredictions.append((
                    self.currentImageId,
                    int(pred[5]),  # class
                    pred[4],       # confidence
                    pred[0], pred[1], pred[2], pred[3]  # box
                ))

        # 處理標籤
        if labels is not None and len(labels) > 0:
            for label in labels:
                self.allLabels.append((
                    self.currentImageId,
                    int(label[0]),  # class
                    label[1], label[2], label[3], label[4]  # box (may need conversion)
                ))

        self.currentImageId += 1

    def compute(self) -> Dict[str, float]:
        """
        計算所有指標

        返回:
            指標字典
        """
        if len(self.allPredictions) == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'mAP50': 0.0,
                'mAP50_95': 0.0,
                'f1': 0.0
            }

        # 按類別組織預測和標籤
        predsByClass = {i: [] for i in range(self.numClasses)}
        labelsByImageClass = {}

        for pred in self.allPredictions:
            imgId, classId, conf, x1, y1, x2, y2 = pred
            if classId < self.numClasses:
                predsByClass[classId].append((imgId, conf, x1, y1, x2, y2))

        for label in self.allLabels:
            imgId, classId, x1, y1, x2, y2 = label
            key = (imgId, classId)
            if key not in labelsByImageClass:
                labelsByImageClass[key] = []
            labelsByImageClass[key].append((x1, y1, x2, y2))

        # 計算每個類別的 AP
        aps = np.zeros((self.numClasses, len(self.iouThresholds)))
        precisions = []
        recalls = []

        for classId in range(self.numClasses):
            classPreds = predsByClass[classId]

            if len(classPreds) == 0:
                continue

            # 按信心排序
            classPreds = sorted(classPreds, key=lambda x: x[1], reverse=True)

            # 計算 TP/FP
            for iouIdx, iouThresh in enumerate(self.iouThresholds):
                tps = []
                fps = []
                matched = {}

                for imgId, conf, px1, py1, px2, py2 in classPreds:
                    key = (imgId, classId)
                    labels = labelsByImageClass.get(key, [])

                    if len(labels) == 0:
                        fps.append(1)
                        tps.append(0)
                        continue

                    # 計算與所有標籤的 IoU
                    predBox = np.array([[px1, py1, px2, py2]])
                    labelBoxes = np.array(labels)

                    if labelBoxes.shape[1] == 4:
                        ious = computeIou(predBox, labelBoxes)[0]
                    else:
                        ious = np.zeros(len(labels))

                    # 找到最佳匹配
                    bestIdx = np.argmax(ious)
                    bestIou = ious[bestIdx]

                    matchKey = (imgId, classId, bestIdx)
                    if bestIou >= iouThresh and matchKey not in matched:
                        tps.append(1)
                        fps.append(0)
                        matched[matchKey] = True
                    else:
                        fps.append(1)
                        tps.append(0)

                # 計算累積 TP/FP
                tps = np.cumsum(tps)
                fps = np.cumsum(fps)

                # 計算所有該類別的標籤數量
                numLabels = sum(
                    len(labelsByImageClass.get((imgId, classId), []))
                    for imgId in set(p[0] for p in classPreds)
                )
                numLabels = max(1, sum(
                    len(v) for k, v in labelsByImageClass.items() if k[1] == classId
                ))

                # 計算 precision 和 recall
                if len(tps) > 0:
                    precision = tps / (tps + fps + 1e-7)
                    recall = tps / (numLabels + 1e-7)

                    # 計算 AP
                    aps[classId, iouIdx] = computeAp(recall, precision)

                    if iouIdx == 0:  # IoU = 0.5
                        if len(precision) > 0:
                            precisions.append(precision[-1])
                            recalls.append(recall[-1])

        # 計算最終指標
        mAP50 = np.mean(aps[:, 0])  # IoU = 0.5
        mAP50_95 = np.mean(aps)     # IoU = 0.5:0.95

        meanPrecision = np.mean(precisions) if precisions else 0.0
        meanRecall = np.mean(recalls) if recalls else 0.0
        f1 = 2 * meanPrecision * meanRecall / (meanPrecision + meanRecall + 1e-7)

        return {
            'precision': meanPrecision,
            'recall': meanRecall,
            'mAP50': mAP50,
            'mAP50_95': mAP50_95,
            'f1': f1
        }

    def reset(self) -> None:
        """重置累積的資料"""
        self.allPredictions = []
        self.allLabels = []
        self.currentImageId = 0


def apPerClass(
    tpList: List[np.ndarray],
    confList: List[np.ndarray],
    predClsList: List[np.ndarray],
    targetClsList: List[np.ndarray],
    numClasses: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    計算每個類別的 AP

    參數:
        tpList: 每張圖的 TP 陣列列表
        confList: 每張圖的信心陣列列表
        predClsList: 每張圖的預測類別列表
        targetClsList: 每張圖的目標類別列表
        numClasses: 類別數量

    返回:
        (精確率, 召回率, AP, F1, 唯一類別)
    """
    # 合併所有圖片的結果
    tp = np.concatenate(tpList)
    conf = np.concatenate(confList)
    predCls = np.concatenate(predClsList)
    targetCls = np.concatenate(targetClsList)

    # 取得唯一類別
    uniqueClasses = np.unique(targetCls)
    numUniqueClasses = len(uniqueClasses)

    # 初始化結果陣列
    ap = np.zeros(numUniqueClasses)
    precision = np.zeros(numUniqueClasses)
    recall = np.zeros(numUniqueClasses)

    for i, cls in enumerate(uniqueClasses):
        # 取得這個類別的預測
        clsMask = predCls == cls
        numPred = clsMask.sum()
        numTarget = (targetCls == cls).sum()

        if numPred == 0 or numTarget == 0:
            continue

        # 按信心排序
        sortIdx = np.argsort(-conf[clsMask])
        tpCls = tp[clsMask][sortIdx]

        # 累積 TP
        tpCumsum = np.cumsum(tpCls)
        fpCumsum = np.cumsum(1 - tpCls)

        # 計算 precision 和 recall
        recallCurve = tpCumsum / numTarget
        precisionCurve = tpCumsum / (tpCumsum + fpCumsum)

        # 計算 AP
        ap[i] = computeAp(recallCurve, precisionCurve)

        # 取最後的 precision 和 recall
        precision[i] = precisionCurve[-1] if len(precisionCurve) > 0 else 0
        recall[i] = recallCurve[-1] if len(recallCurve) > 0 else 0

    # 計算 F1
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    return precision, recall, ap, f1, uniqueClasses.astype(int)


def fitnessScore(metrics: Dict[str, float]) -> float:
    """
    計算模型的適應度分數

    用於選擇最佳模型。權重可以根據應用調整。

    參數:
        metrics: 指標字典

    返回:
        適應度分數
    """
    # 權重: [mAP@0.5 的權重, mAP@0.5:0.95 的權重]
    weights = [0.1, 0.9]

    mAP50 = metrics.get('mAP50', 0.0)
    mAP50_95 = metrics.get('mAP50_95', 0.0)

    return weights[0] * mAP50 + weights[1] * mAP50_95
