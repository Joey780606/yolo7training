"""
損失函數模組 (Loss Functions Module)

此模組實現了 YOLOv7 的損失函數，包含三個主要組件：
1. 邊界框損失 (Box Loss): 使用 CIoU 衡量預測框與真實框的差異
2. 物件性損失 (Objectness Loss): 判斷格子內是否有物體
3. 分類損失 (Classification Loss): 物體屬於哪個類別

損失計算是 YOLO 訓練的核心，理解損失函數有助於調試和改進模型。

總損失 = λ_box * box_loss + λ_obj * obj_loss + λ_cls * cls_loss
"""

import math
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def smoothBCE(eps: float = 0.1) -> Tuple[float, float]:
    """
    計算標籤平滑的 BCE 目標值

    標籤平滑是一種正則化技術，可以防止模型過於自信。
    原本的 0/1 標籤會被替換為接近但不等於 0/1 的值。

    參數:
        eps: 平滑係數

    返回:
        (正樣本目標值, 負樣本目標值)
    """
    # 正樣本: 1.0 - eps/2 (例如 0.95)
    # 負樣本: eps/2 (例如 0.05)
    return 1.0 - eps * 0.5, eps * 0.5


class FocalLoss(nn.Module):
    """
    Focal Loss

    Focal Loss 是為了解決類別不平衡問題而設計的損失函數。
    它通過降低「容易」樣本的權重，讓模型更專注於「困難」樣本。

    公式: FL(p) = -α * (1-p)^γ * log(p)

    其中:
    - p: 預測機率
    - γ (gamma): 聚焦參數，γ > 0 會降低容易樣本的影響
    - α (alpha): 平衡正負樣本的權重

    當 γ = 0 時，Focal Loss 退化為標準交叉熵損失。
    """

    def __init__(self, baseLoss: nn.Module, gamma: float = 1.5, alpha: float = 0.25):
        """
        初始化 Focal Loss

        參數:
            baseLoss: 基礎損失函數（通常是 BCEWithLogitsLoss）
            gamma: 聚焦參數，越大則越專注於困難樣本
            alpha: 正樣本權重
        """
        super().__init__()
        self.baseLoss = baseLoss
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        計算 Focal Loss

        參數:
            predictions: 預測值（未經過 sigmoid）
            targets: 目標值

        返回:
            Focal Loss 值
        """
        # 計算基礎 BCE 損失
        loss = self.baseLoss(predictions, targets)

        # 計算預測機率
        predProb = torch.sigmoid(predictions)

        # 計算 p_t（正確分類的機率）
        pT = predProb * targets + (1 - predProb) * (1 - targets)

        # 計算 alpha_t
        alphaT = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # 應用 Focal 調製
        focalWeight = alphaT * (1 - pT) ** self.gamma

        return (loss * focalWeight).mean()


class ComputeLoss:
    """
    YOLO 損失計算器

    這個類別負責計算 YOLOv7 的所有損失組件。
    它處理錨點框匹配、目標分配和損失計算。

    YOLOv7 的損失計算流程:
    1. 為每個預測找到最匹配的真實目標（基於 IoU）
    2. 計算邊界框回歸損失（CIoU Loss）
    3. 計算物件性損失（BCE Loss）
    4. 計算分類損失（BCE Loss）
    5. 加權合併所有損失

    屬性:
        hyperParams: 超參數字典
        device: 計算裝置
        numClasses: 類別數量
        anchors: 錨點框
    """

    def __init__(self, model: nn.Module, hyperParams: Dict[str, Any] = None):
        """
        初始化損失計算器

        參數:
            model: YOLO 模型
            hyperParams: 超參數字典
        """
        self.hyperParams = hyperParams or {}

        # 取得模型的檢測頭
        if hasattr(model, 'head'):
            detect = model.head.detect
        elif hasattr(model, 'module'):
            detect = model.module.head.detect
        else:
            detect = model

        self.device = next(model.parameters()).device

        # 從檢測頭取得配置
        self.numClasses = detect.numClasses
        self.numLayers = detect.numDetectLayers
        self.numAnchors = detect.numAnchorsPerScale
        self.strides = detect.strides

        # 取得錨點框（已經根據 stride 縮放過）
        self.anchors = detect.anchorsOriginal / self.strides.view(-1, 1, 1)
        self.anchors = self.anchors.to(self.device)

        # 損失函數權重（來自超參數或使用預設值）
        self.boxWeight = self.hyperParams.get('box', 0.05)
        self.objWeight = self.hyperParams.get('obj', 0.7)
        self.clsWeight = self.hyperParams.get('cls', 0.3)

        # 平衡不同尺度的損失
        # 較小的特徵圖（檢測大物體）權重較低
        self.balanceWeights = [4.0, 1.0, 0.4]

        # 標籤平滑
        smoothParams = smoothBCE(eps=self.hyperParams.get('label_smoothing', 0.0))
        self.positiveTarget, self.negativeTarget = smoothParams

        # BCE 損失函數
        self.bceCls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=self.device))
        self.bceObj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=self.device))

        # 是否使用 Focal Loss
        if self.hyperParams.get('focal_loss', False):
            gamma = self.hyperParams.get('focal_gamma', 1.5)
            self.bceCls = FocalLoss(self.bceCls, gamma=gamma)
            self.bceObj = FocalLoss(self.bceObj, gamma=gamma)

        # 錨點框閾值（用於正負樣本分配）
        self.anchorThreshold = self.hyperParams.get('anchor_t', 4.0)

        # 梯度累積的批次大小
        self.gradientAccum = 1.0

    def __call__(
        self,
        predictions: List[torch.Tensor],
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        計算總損失

        參數:
            predictions: 各尺度的預測
                        列表中每個元素形狀: [batch, numAnchors, H, W, 5 + numClasses]
            targets: 真實標籤 [numTargets, 6]
                    每行: [batchIdx, classId, x, y, w, h]

        返回:
            (總損失, 損失字典)
        """
        # 初始化損失
        lossBox = torch.zeros(1, device=self.device)
        lossObj = torch.zeros(1, device=self.device)
        lossCls = torch.zeros(1, device=self.device)

        # 建立目標（為每個預測分配正負樣本）
        targetCls, targetBox, targetIndices, targetAnchors = self.buildTargets(predictions, targets)

        # 計算每個尺度的損失
        for layerIdx, prediction in enumerate(predictions):
            batchSize = prediction.shape[0]

            # 取得這一層匹配到的目標
            batchIdxs, anchorIdxs, gridYs, gridXs = targetIndices[layerIdx]

            # 物件性目標（預設全為 0，即背景）
            objTarget = torch.zeros_like(prediction[..., 4])

            numTargets = batchIdxs.shape[0]

            if numTargets > 0:
                # 取得對應的預測
                # prediction[batchIdx, anchorIdx, gridY, gridX] -> [numTargets, 5 + numClasses]
                predSubset = prediction[batchIdxs, anchorIdxs, gridYs, gridXs]

                # ============= 邊界框損失 =============
                # 預測的 xy 偏移和 wh 縮放
                predXY = predSubset[:, :2].sigmoid() * 2 - 0.5
                predWH = (predSubset[:, 2:4].sigmoid() * 2) ** 2 * targetAnchors[layerIdx]

                # 組合成預測框 [x, y, w, h]
                predBox = torch.cat([predXY, predWH], dim=1)

                # 計算 CIoU
                ciou = self.computeCiou(predBox, targetBox[layerIdx])

                # 邊界框損失 = 1 - CIoU
                lossBox += (1.0 - ciou).mean()

                # 設定物件性目標（使用 IoU 作為軟標籤）
                objTarget[batchIdxs, anchorIdxs, gridYs, gridXs] = ciou.detach().clamp(0).type(objTarget.dtype)

                # ============= 分類損失 =============
                if self.numClasses > 1:
                    # 建立分類目標
                    clsTarget = torch.zeros_like(predSubset[:, 5:])
                    clsTarget[range(numTargets), targetCls[layerIdx]] = self.positiveTarget

                    lossCls += self.bceCls(predSubset[:, 5:], clsTarget)

            # ============= 物件性損失 =============
            lossObj += self.bceObj(prediction[..., 4], objTarget) * self.balanceWeights[layerIdx]

        # 應用權重並合併損失
        lossBox *= self.boxWeight
        lossObj *= self.objWeight
        lossCls *= self.clsWeight

        # 根據批次大小縮放
        batchSize = predictions[0].shape[0]
        totalLoss = (lossBox + lossObj + lossCls) * batchSize

        # 返回損失字典供記錄
        lossDict = {
            'box': lossBox.item(),
            'obj': lossObj.item(),
            'cls': lossCls.item(),
            'total': totalLoss.item()
        }

        return totalLoss, lossDict

    def buildTargets(
        self,
        predictions: List[torch.Tensor],
        targets: torch.Tensor
    ) -> Tuple[List, List, List, List]:
        """
        建立訓練目標

        為每個預測位置分配正負樣本。YOLOv7 使用以下策略：
        1. 對於每個真實目標，找到最匹配的錨點框
        2. 如果長寬比在閾值範圍內，則為正樣本
        3. 考慮相鄰網格單元（增加正樣本數量）

        參數:
            predictions: 各尺度的預測
            targets: 真實標籤 [numTargets, 6]

        返回:
            (類別列表, 邊界框列表, 索引列表, 錨點框列表)
        """
        numTargets = targets.shape[0]
        targetCls, targetBox, indices, anchors = [], [], [], []

        # 擴展目標以匹配錨點框
        # targets: [numTargets, 6] -> [numAnchors, numTargets, 7]
        # 新增一個維度存儲錨點索引
        gain = torch.ones(7, device=targets.device)

        # 錨點索引 [numAnchors, 1] -> [[0], [1], [2]] 對應 3 個錨點
        anchorIndices = torch.arange(self.numAnchors, device=targets.device).float()
        anchorIndices = anchorIndices.view(self.numAnchors, 1).repeat(1, numTargets)

        # 擴展目標: [numAnchors, numTargets, 7]
        # 最後一維: [batchIdx, classId, x, y, w, h, anchorIdx]
        targetsExpanded = torch.cat([
            targets.repeat(self.numAnchors, 1, 1),
            anchorIndices[:, :, None]
        ], dim=2)

        # 相鄰網格偏移（用於增加正樣本）
        offsets = torch.tensor([
            [0, 0],
            [1, 0], [0, 1], [-1, 0], [0, -1],  # 上下左右
        ], device=targets.device).float() * 0.5

        # 處理每個檢測層
        for layerIdx in range(self.numLayers):
            # 這一層的錨點框
            layerAnchors = self.anchors[layerIdx]

            # 取得這一層的網格大小
            _, _, gridH, gridW, _ = predictions[layerIdx].shape

            # 更新增益（用於將正規化座標轉換為網格座標）
            gain[2:6] = torch.tensor([gridW, gridH, gridW, gridH], device=targets.device)

            # 將目標座標轉換為網格座標
            t = targetsExpanded * gain

            if numTargets > 0:
                # 計算錨點框與目標的長寬比
                # 目標寬高 / 錨點框寬高
                ratio = t[:, :, 4:6] / layerAnchors[:, None]

                # 取最大比例（考慮正反比）
                maxRatio = torch.max(ratio, 1.0 / ratio).max(dim=2)[0]

                # 過濾長寬比超過閾值的匹配
                validMask = maxRatio < self.anchorThreshold

                # 取得有效的目標
                t = t[validMask]

                # 計算網格位置
                gridXY = t[:, 2:4]
                gridXYInverse = gain[[2, 3]] - gridXY

                # 判斷是否接近相鄰網格（用於增加正樣本）
                # j: 接近右邊界, k: 接近下邊界
                j = (gridXY % 1 < 0.5) & (gridXY > 1)
                k = (gridXYInverse % 1 < 0.5) & (gridXYInverse > 1)

                # 選擇偏移
                # 預設包含中心位置 [0, 0]
                selectedOffsets = torch.zeros_like(gridXY)[None].repeat(5, 1, 1)
                for i, (offX, offY) in enumerate(offsets):
                    selectedOffsets[i] = torch.stack([
                        torch.full_like(gridXY[:, 0], offX),
                        torch.full_like(gridXY[:, 1], offY)
                    ], dim=1)

                # 合併有效的偏移位置
                offsetMask = torch.stack([
                    torch.ones_like(j[:, 0]),  # 中心位置總是有效
                    j[:, 0], k[:, 1], ~j[:, 0], ~k[:, 1]
                ])

                # 擴展目標以包含偏移位置
                t = t.repeat(5, 1, 1)[offsetMask]
                offsets_selected = selectedOffsets[offsetMask]

            else:
                t = targetsExpanded[0]
                offsets_selected = torch.zeros((0, 2), device=targets.device)

            # 提取最終的目標資訊
            batchIdx = t[:, 0].long()
            classIdx = t[:, 1].long()
            gridXY = t[:, 2:4]
            gridWH = t[:, 4:6]
            anchorIdx = t[:, 6].long()

            # 計算網格索引（考慮偏移）
            gridIJ = (gridXY - offsets_selected).long()
            gridI = gridIJ[:, 0].clamp(0, gridW - 1)  # x
            gridJ = gridIJ[:, 1].clamp(0, gridH - 1)  # y

            # 儲存結果
            indices.append((batchIdx, anchorIdx, gridJ, gridI))
            targetBox.append(torch.cat([gridXY - gridIJ, gridWH], dim=1))
            anchors.append(layerAnchors[anchorIdx])
            targetCls.append(classIdx)

        return targetCls, targetBox, indices, anchors

    def computeCiou(
        self,
        predBox: torch.Tensor,
        targetBox: torch.Tensor,
        eps: float = 1e-7
    ) -> torch.Tensor:
        """
        計算 CIoU (Complete IoU)

        CIoU 考慮了三個因素：
        1. 重疊面積 (IoU)
        2. 中心點距離
        3. 長寬比一致性

        參數:
            predBox: 預測框 [N, 4] (x, y, w, h)
            targetBox: 目標框 [N, 4] (x, y, w, h)
            eps: 防止除零

        返回:
            CIoU 值 [N]
        """
        # 轉換為角點座標
        predX1 = predBox[:, 0] - predBox[:, 2] / 2
        predY1 = predBox[:, 1] - predBox[:, 3] / 2
        predX2 = predBox[:, 0] + predBox[:, 2] / 2
        predY2 = predBox[:, 1] + predBox[:, 3] / 2

        targetX1 = targetBox[:, 0] - targetBox[:, 2] / 2
        targetY1 = targetBox[:, 1] - targetBox[:, 3] / 2
        targetX2 = targetBox[:, 0] + targetBox[:, 2] / 2
        targetY2 = targetBox[:, 1] + targetBox[:, 3] / 2

        # 計算交集
        interX1 = torch.max(predX1, targetX1)
        interY1 = torch.max(predY1, targetY1)
        interX2 = torch.min(predX2, targetX2)
        interY2 = torch.min(predY2, targetY2)
        interArea = (interX2 - interX1).clamp(0) * (interY2 - interY1).clamp(0)

        # 計算各自面積
        predArea = predBox[:, 2] * predBox[:, 3]
        targetArea = targetBox[:, 2] * targetBox[:, 3]
        unionArea = predArea + targetArea - interArea + eps

        # IoU
        iou = interArea / unionArea

        # 最小包圍框
        convexX1 = torch.min(predX1, targetX1)
        convexY1 = torch.min(predY1, targetY1)
        convexX2 = torch.max(predX2, targetX2)
        convexY2 = torch.max(predY2, targetY2)

        # 對角線距離的平方
        convexDiag = (convexX2 - convexX1) ** 2 + (convexY2 - convexY1) ** 2 + eps

        # 中心點距離的平方
        centerDist = (predBox[:, 0] - targetBox[:, 0]) ** 2 + (predBox[:, 1] - targetBox[:, 1]) ** 2

        # 長寬比一致性
        v = (4 / math.pi ** 2) * torch.pow(
            torch.atan(targetBox[:, 2] / (targetBox[:, 3] + eps)) -
            torch.atan(predBox[:, 2] / (predBox[:, 3] + eps)),
            2
        )

        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)

        # CIoU
        ciou = iou - centerDist / convexDiag - alpha * v

        return ciou
