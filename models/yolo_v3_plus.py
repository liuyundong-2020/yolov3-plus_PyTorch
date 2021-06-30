import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Conv, SPP, BottleneckCSP, UpSample
from backbone import *
import numpy as np
import tools

class YOLOv3Plus(nn.Module):
    def __init__(self, 
                 device, 
                 img_size=640, 
                 num_classes=80, 
                 trainable=False, 
                 conf_thresh=0.001, 
                 nms_thresh=0.60, 
                 anchor_size=None, 
                 hr=False, 
                 bk='cd53'):
        super(YOLOv3Plus, self).__init__()
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = [8, 16, 32]
        self.bk = bk
        self.anchor_size = torch.tensor(anchor_size).view(len(self.stride), len(anchor_size) // 3, 2)
        self.num_anchors = self.anchor_size.size(1)
        self.grid_cell, self.stride_tensor, self.anchors_wh = self.create_grid(img_size)

        # backbone
        if self.bk == 'cd53':
            print('backbone: CSPDarkNet-53 ...')
            self.backbone = cspdarknet53(pretrained=trainable, hr=hr)
            c3, c4, c5 = 256, 512, 1024

        # neck
        self.neck = nn.Sequential(
            SPP(c5, c5, e=0.5),
            BottleneckCSP(c5, c5, n=3, shortcut=False)
        )

         # head
        self.head_conv_0 = Conv(c5, c5//2, k=1)  # 10
        self.head_upsample_0 = UpSample(scale_factor=2)
        self.head_csp_0 = BottleneckCSP(c4 + c5//2, c4, n=3, shortcut=False)

        # P3/8-small
        self.head_conv_1 = Conv(c4, c4//2, k=1)  # 14
        self.head_upsample_1 = UpSample(scale_factor=2)
        self.head_csp_1 = BottleneckCSP(c3 + c4//2, c3, n=3, shortcut=False)

        # P4/16-medium
        self.head_conv_2 = Conv(c3, c3, k=3, p=1, s=2)
        self.head_csp_2 = BottleneckCSP(c3 + c4//2, c4, n=3, shortcut=False)

        # P8/32-large
        self.head_conv_3 = Conv(c4, c4, k=3, p=1, s=2)
        self.head_csp_3 = BottleneckCSP(c4 + c5//2, c5, n=3, shortcut=False)

        # det conv
        self.head_det_1 = nn.Conv2d(c3, self.num_anchors * (1 + self.num_classes + 4), 1)
        self.head_det_2 = nn.Conv2d(c4, self.num_anchors * (1 + self.num_classes + 4), 1)
        self.head_det_3 = nn.Conv2d(c5, self.num_anchors * (1 + self.num_classes + 4), 1)


    def create_grid(self, img_size):
        total_grid_xy = []
        total_stride = []
        total_anchor_wh = []
        w, h = img_size, img_size
        for ind, s in enumerate(self.stride):
            # generate grid cells
            ws, hs = w // s, h // s
            grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
            grid_xy = grid_xy.view(1, hs*ws, 1, 2)

            # generate stride tensor
            stride_tensor = torch.ones([1, hs*ws, self.num_anchors, 2]) * s

            # generate anchor_wh tensor
            anchor_wh = self.anchor_size[ind].repeat(hs*ws, 1, 1)

            total_grid_xy.append(grid_xy)
            total_stride.append(stride_tensor)
            total_anchor_wh.append(anchor_wh)

        total_grid_xy = torch.cat(total_grid_xy, dim=1).to(self.device)
        total_stride = torch.cat(total_stride, dim=1).to(self.device)
        total_anchor_wh = torch.cat(total_anchor_wh, dim=0).to(self.device).unsqueeze(0)

        return total_grid_xy, total_stride, total_anchor_wh


    def set_grid(self, img_size):
        self.img_size = img_size
        self.grid_cell, self.stride_tensor, self.anchors_wh = self.create_grid(img_size)


    def decode_xywh(self, reg_pred):
        """
            Input:
                reg_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [x, y, w, h]
        """
        # b_x = sigmoid(tx) + gride_x
        # b_y = sigmoid(ty) + gride_y
        B, HW, ab_n, _ = reg_pred.size()
        c_xy_pred = (torch.sigmoid(reg_pred[:, :, :, :2]) + self.grid_cell) * self.stride_tensor
        # b_w = anchor_w * exp(tw)
        # b_h = anchor_h * exp(th)
        b_wh_pred = torch.exp(reg_pred[:, :, :, 2:]) * self.anchors_wh
        # [B, H*W, anchor_n, 4] -> [B, H*W*anchor_n, 4]
        xywh_pred = torch.cat([c_xy_pred, b_wh_pred], -1).view(B, HW*ab_n, 4)

        return xywh_pred


    def decode_boxes(self, reg_pred):
        """
            Input:
                reg_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                x1y1x2y2_pred : [B, H*W, anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # [B, H*W*anchor_n, 4]
        xywh_pred = self.decode_xywh(reg_pred)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        x1y1x2y2_pred = torch.zeros_like(xywh_pred)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)
        
        return x1y1x2y2_pred


    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)                 # the size of bbox
        order = scores.argsort()[::-1]                        # sort bounding boxes by decreasing order

        keep = []                                             # store the final bounding boxes
        while order.size > 0:
            i = order[0]                                      #the index of the bbox with highest confidence
            keep.append(i)                                    #save it to keep
            # compute iou
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, bboxes, scores):
        """
        bboxes: (HxW, 4), bsize = 1
        scores: (HxW, num_classes), bsize = 1
        """

        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds


    def forward(self, x, targets=None):
        # backbone
        c3, c4, c5 = self.backbone(x)

        # neck
        c5 = self.neck(c5)

        # FPN + PAN
        # head
        c6 = self.head_conv_0(c5)
        c7 = self.head_upsample_0(c6)   # s32->s16
        c8 = torch.cat([c7, c4], dim=1)
        c9 = self.head_csp_0(c8)
        # P3/8
        c10 = self.head_conv_1(c9)
        c11 = self.head_upsample_1(c10)   # s16->s8
        c12 = torch.cat([c11, c3], dim=1)
        c13 = self.head_csp_1(c12)  # to det
        # p4/16
        c14 = self.head_conv_2(c13)
        c15 = torch.cat([c14, c10], dim=1)
        c16 = self.head_csp_2(c15)  # to det
        # p5/32
        c17 = self.head_conv_3(c16)
        c18 = torch.cat([c17, c6], dim=1)
        c19 = self.head_csp_3(c18)  # to det

        # det
        pred_s = self.head_det_1(c13)
        pred_m = self.head_det_2(c16)
        pred_l = self.head_det_3(c19)

        preds = [pred_s, pred_m, pred_l]
        total_obj_pred = []
        total_cls_pred = []
        total_reg_pred = []
        B = HW = 0
        for pred in preds:
            B_, abC_, H_, W_ = pred.size()

            # [B, anchor_n * C, H, W] -> [B, H, W, anchor_n * C] -> [B, H*W, anchor_n*C]
            pred = pred.permute(0, 2, 3, 1).reshape(B_, H_*W_, abC_)

            # Divide prediction to obj_pred, xywh_pred and cls_pred   
            # [B, H*W*anchor_n, 1]
            obj_pred = pred[:, :, :1 * self.num_anchors].reshape(B_, H_*W_*self.num_anchors, -1)
            # [B, H*W*anchor_n, num_cls]
            cls_pred = pred[:, :, 1 * self.num_anchors : (1 + self.num_classes) * self.num_anchors].reshape(B_, H_*W_*self.num_anchors, -1)
            # [B, H*W*anchor_n, 4]
            reg_pred = pred[:, :, (1 + self.num_classes) * self.num_anchors:].reshape(B_, H_*W_, self.num_anchors, -1)

            total_obj_pred.append(obj_pred)
            total_cls_pred.append(cls_pred)
            total_reg_pred.append(reg_pred)
            B = B_
            HW += H_*W_
        
        obj_pred = torch.cat(total_obj_pred, dim=1)
        cls_pred = torch.cat(total_cls_pred, dim=1)
        reg_pred = torch.cat(total_reg_pred, dim=1)
        
        # train
        if self.trainable:           
            # decode bbox
            box_pred = (self.decode_boxes(reg_pred) / self.img_size)

            # loss
            obj_loss, cls_loss, reg_loss, total_loss = tools.loss(pred_obj=obj_pred,
                                                                  pred_cls=cls_pred,
                                                                  pred_box=box_pred,
                                                                  targets=targets)

            return obj_loss, cls_loss, reg_loss, total_loss

        # test
        else:
            with torch.no_grad():
                # batch size = 1
                # [B, H*W*num_anchor, 1] -> [H*W*num_anchor, 1]
                obj_pred = torch.sigmoid(obj_pred)[0]
                # [B, H*W*num_anchor, 4] -> [H*W*num_anchor, 4]
                bboxes = torch.clamp((self.decode_boxes(reg_pred) / self.img_size)[0], 0., 1.)
                # [B, H*W*num_anchor, C] -> [H*W*num_anchor, C], 
                scores = torch.softmax(cls_pred[0, :, :], dim=1) * obj_pred

                # 将预测放在cpu处理上，以便进行后处理
                scores = scores.to('cpu').numpy()
                bboxes = bboxes.to('cpu').numpy()

                # 后处理
                bboxes, scores, cls_inds = self.postprocess(bboxes, scores)

                return bboxes, scores, cls_inds
