import numpy as np
import random
import torch
import os
from common import utils
from common.logger import  AverageMeter
from common.evaluation import Evaluator

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def Compute_iou(model, dataloader, nshot):


    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)

        qry_rgb = batch['query_img']
        qry_msk = batch['query_mask']

        # === K-shot support ===
        sup_imgs = batch['support_imgs']  # [B, K, 3, H, W]
        sup_msks_ = batch['support_masks']  # [B, K, H, W]

        # 将 K 个 support 拆成 list，符合模型 forward 的格式
        sup_rgbs = [sup_imgs[:, k] for k in range(sup_imgs.shape[1])]
        sup_msks = [sup_msks_[:, k] for k in range(sup_msks_.shape[1])]

        # === 推理 ===
        pred = model(sup_rgbs, sup_msks, qry_rgb, qry_msk, training=False)[0]

        pred_mask = torch.argmax(pred, dim=1)

        assert pred_mask.size() == batch['query_mask'].size()
        # Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou

class mIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return np.nanmean(iu[1:])
