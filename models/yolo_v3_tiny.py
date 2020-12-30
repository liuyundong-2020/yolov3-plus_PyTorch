import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Conv, SPP, BottleneckCSP, UpSample
from backbone import *
import numpy as np
import tools


class YOLOv3Tiny(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.50, anchor_size=None, hr=False, backbone='d-tiny', ciou=False):
        super(YOLOv3Tiny, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.bk = backbone
        self.ciou = ciou
        self.stride = [8, 16, 32]
        self.anchor_size = torch.tensor(anchor_size).view(3, len(anchor_size) // 3, 2)
        self.anchor_number = self.anchor_size.size(1)

        self.grid_cell, self.stride_tensor, self.all_anchors_wh = self.create_grid(input_size)
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=device).float()

        if self.bk == 'd-tiny':
            # backbone darknet-tiny
            print('Use backbone: d-tiny')
            self.backbone = darknet_tiny(pretrained=trainable, hr=hr)
        elif self.bk == 'csp-tiny':
            # use cspdarknet_tiny as backbone
            print('Use backbone: csp-tiny')
            self.backbone = cspdarknet_tiny(pretrained=trainable, hr=hr)
        else:
            print("For YOLOv3Tiny, we only support <d-tiny, csp-tiny> as our backbone !!")
            exit(0)

        # SPP
        self.spp = nn.Sequential(
            Conv(512, 256, k=1),
            SPP(),
            Conv(256*4, 512, k=1)
        )

        # head
        self.head_conv_0 = Conv(512, 256, k=1)  # 10
        self.head_upsample_0 = UpSample(scale_factor=2)
        self.head_csp_0 = Conv(256 + 256, 256, k=1)

        # P3/8-small
        self.head_conv_1 = Conv(256, 128, k=1)  # 14
        self.head_upsample_1 = UpSample(scale_factor=2)
        self.head_csp_1 = Conv(128 + 128, 128, k=1)

        # P4/16-medium
        self.head_downsample_2 = nn.MaxPool2d((3, 3), stride=2, padding=1)
        self.head_csp_2 = Conv(128 + 128, 256, k=1)

        # P8/32-large
        self.head_downsample_3 = nn.MaxPool2d((3, 3), stride=2, padding=1)
        self.head_csp_3 = Conv(256 + 256, 512, k=1)

        # det conv
        self.head_det_1 = nn.Conv2d(128, self.anchor_number * (1 + self.num_classes + 4), 1)
        self.head_det_2 = nn.Conv2d(256, self.anchor_number * (1 + self.num_classes + 4), 1)
        self.head_det_3 = nn.Conv2d(512, self.anchor_number * (1 + self.num_classes + 4), 1)


    def create_grid(self, input_size):
        total_grid_xy = []
        total_stride = []
        total_anchor_wh = []
        w, h = input_size[1], input_size[0]
        for ind, s in enumerate(self.stride):
            # generate grid cells
            ws, hs = w // s, h // s
            grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
            grid_xy = grid_xy.view(1, hs*ws, 1, 2)

            # generate stride tensor
            stride_tensor = torch.ones([1, hs*ws, self.anchor_number, 2]) * s

            # generate anchor_wh tensor
            anchor_wh = self.anchor_size[ind].repeat(hs*ws, 1, 1)

            total_grid_xy.append(grid_xy)
            total_stride.append(stride_tensor)
            total_anchor_wh.append(anchor_wh)

        total_grid_xy = torch.cat(total_grid_xy, dim=1).to(self.device)
        total_stride = torch.cat(total_stride, dim=1).to(self.device)
        total_anchor_wh = torch.cat(total_anchor_wh, dim=0).to(self.device).unsqueeze(0)

        return total_grid_xy, total_stride, total_anchor_wh


    def set_grid(self, input_size):
        self.grid_cell, self.stride_tensor, self.all_anchors_wh = self.create_grid(input_size)
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=self.device).float()


    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [x, y, w, h]
        """
        # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
        B, HW, ab_n, _ = txtytwth_pred.size()
        c_xy_pred = (torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_cell) * self.stride_tensor
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        b_wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.all_anchors_wh
        # [B, H*W, anchor_n, 4] -> [B, H*W*anchor_n, 4]
        xywh_pred = torch.cat([c_xy_pred, b_wh_pred], -1).view(B, HW*ab_n, 4)

        return xywh_pred


    def decode_boxes(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                x1y1x2y2_pred : [B, H*W, anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # [B, H*W*anchor_n, 4]
        xywh_pred = self.decode_xywh(txtytwth_pred)

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
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, all_local, all_conf, exchange=True, im_shape=None):
        """
        bbox_pred: (HxW*anchor_n, 4), bsize = 1
        prob_pred: (HxW*anchor_n, num_classes), bsize = 1
        """
        bbox_pred = all_local
        prob_pred = all_conf

        cls_inds = np.argmax(prob_pred, axis=1)
        prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), cls_inds)]
        scores = prob_pred.copy()
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bbox_pred), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bbox_pred[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        if im_shape != None:
            # clip
            bbox_pred = self.clip_boxes(bbox_pred, im_shape)

        return bbox_pred, scores, cls_inds


    def forward(self, x, target=None):
        # backbone
        c3, c4, c5 = self.backbone(x)

        # neck
        c5 = self.spp(c5)

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
        c14 = self.head_downsample_2(c13)
        c15 = torch.cat([c14, c10], dim=1)
        c16 = self.head_csp_2(c15)  # to det
        # p5/32
        c17 = self.head_downsample_3(c16)
        c18 = torch.cat([c17, c6], dim=1)
        c19 = self.head_csp_3(c18)  # to det

        # det
        pred_s = self.head_det_1(c13)
        pred_m = self.head_det_2(c16)
        pred_l = self.head_det_3(c19)

        preds = [pred_s, pred_m, pred_l]
        total_conf_pred = []
        total_cls_pred = []
        total_txtytwth_pred = []
        B = HW = 0
        for pred in preds:
            B_, abC_, H_, W_ = pred.size()

            # [B, anchor_n * C, H, W] -> [B, H, W, anchor_n * C] -> [B, H*W, anchor_n*C]
            pred = pred.permute(0, 2, 3, 1).contiguous().view(B_, H_*W_, abC_)

            # Divide prediction to obj_pred, xywh_pred and cls_pred   
            # [B, H*W*anchor_n, 1]
            conf_pred = pred[:, :, :1 * self.anchor_number].contiguous().view(B_, H_*W_*self.anchor_number, 1)
            # [B, H*W*anchor_n, num_cls]
            cls_pred = pred[:, :, 1 * self.anchor_number : (1 + self.num_classes) * self.anchor_number].contiguous().view(B_, H_*W_*self.anchor_number, self.num_classes)
            # [B, H*W*anchor_n, 4]
            txtytwth_pred = pred[:, :, (1 + self.num_classes) * self.anchor_number:].contiguous()

            total_conf_pred.append(conf_pred)
            total_cls_pred.append(cls_pred)
            total_txtytwth_pred.append(txtytwth_pred)
            B = B_
            HW += H_*W_
        
        conf_pred = torch.cat(total_conf_pred, 1)
        cls_pred = torch.cat(total_cls_pred, 1)
        txtytwth_pred = torch.cat(total_txtytwth_pred, 1)
        
        # train
        if self.trainable:
            txtytwth_pred = txtytwth_pred.view(B, HW, self.anchor_number, 4)
            
            if self.ciou:
                # use CIoU loss to regress bbox
                x1y1x2y2_pred_ = (self.decode_boxes(txtytwth_pred) / self.scale_torch).view(-1, 4)
                with torch.no_grad():
                    x1y1x2y2_pred = x1y1x2y2_pred_
                
                x1y1x2y2_gt = target[:, :, 7:].view(-1, 4)

                # compute iou and ciou
                iou = tools.iou_score(x1y1x2y2_pred, x1y1x2y2_gt).view(B, -1, 1)
                ciou_pred = tools.CIoU(x1y1x2y2_pred_, x1y1x2y2_gt, batch_size=B)

                # [obj, cls, txtytwth, bbox_weight, x1y1x2y2] -> [conf, obj, cls, bbox_weight]
                target = torch.cat([iou, target[:, :, :3]], dim=2)

                conf_loss, cls_loss, ciou_loss, total_loss = tools.loss_ciou(pred_conf=conf_pred, 
                                                                             pred_cls=cls_pred,
                                                                             pred_ciou=ciou_pred,
                                                                             label=target,
                                                                             num_classes=self.num_classes,
                                                                             obj_loss_f='mse')

                return conf_loss, cls_loss, ciou_loss, total_loss

            else:
                # no CIoU
                # decode bbox, and remember to cancel its grad since we set iou as the label of objectness.
                with torch.no_grad():
                    x1y1x2y2_pred = (self.decode_boxes(txtytwth_pred) / self.scale_torch).view(-1, 4)

                txtytwth_pred = txtytwth_pred.view(B, -1, 4)

                x1y1x2y2_gt = target[:, :, 7:].view(-1, 4)

                # compute iou
                iou = tools.iou_score(x1y1x2y2_pred, x1y1x2y2_gt).view(B, -1, 1)
                # print(iou.min(), iou.max())

                # we set iou between pred bbox and gt bbox as conf label. 
                # [obj, cls, txtytwth, x1y1x2y2] -> [conf, obj, cls, txtytwth]
                target = torch.cat([iou, target[:, :, :7]], dim=2)

                conf_loss, cls_loss, txtytwth_loss, total_loss = tools.loss(pred_conf=conf_pred, pred_cls=cls_pred,
                                                                            pred_txtytwth=txtytwth_pred,
                                                                            label=target,
                                                                            num_classes=self.num_classes,
                                                                            obj_loss_f='mse')

                return conf_loss, cls_loss, txtytwth_loss, total_loss

        # test
        else:
            txtytwth_pred = txtytwth_pred.view(B, HW, self.anchor_number, 4)
            with torch.no_grad():
                # batch size = 1                
                all_obj = torch.sigmoid(conf_pred)[0]           # 0 is because that these is only 1 batch.
                all_bbox = torch.clamp((self.decode_boxes(txtytwth_pred) / self.scale_torch)[0], 0., 1.)
                all_class = (torch.softmax(cls_pred[0, :, :], dim=1) * all_obj)
                # separate box pred and class conf
                all_obj = all_obj.to('cpu').numpy()
                all_class = all_class.to('cpu').numpy()
                all_bbox = all_bbox.to('cpu').numpy()

                bboxes, scores, cls_inds = self.postprocess(all_bbox, all_class)

                # print(len(all_boxes))
                return bboxes, scores, cls_inds

