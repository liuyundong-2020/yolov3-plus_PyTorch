import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import VOC_ROOT, VOC_CLASSES
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from data import config
import numpy as np
import cv2
import tools
import time
from decimal import *


parser = argparse.ArgumentParser(description='YOLO Detection')
parser.add_argument('-v', '--version', default='yolo_v3_spp',
                    help='yolo_v3_spp, tiny_yolo_v3_spp')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument('-size', '--input_size', default=416, type=int, 
                    help='Batch size for training')
parser.add_argument('--trained_model', default='weight/voc/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--visual_threshold', default=0.3, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', action='store_true', default=False, 
                    help='use cuda.')
parser.add_argument('--voc_root', default=VOC_ROOT, 
                    help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, 
                    help="Dummy arg so we can load in Jupyter Notebooks")

args = parser.parse_args()

def test_net(net, device, testset, input_size, thresh, mode='voc'):
    num_images = len(testset)
    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        img_raw = testset.pull_image(index)

        img_tensor, _, h, w, offset, scale = testset.pull_item(index)
        # img_id, annotation = testset.pull_anno(i)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        t0 = time.clock()
        bboxes, scores, cls_inds = net(img_tensor)
        print("detection time used ", Decimal(time.clock()) - Decimal(t0), "s")
        # scale each detection back up to the image
        max_line = max(h, w)
        # map the boxes to input image with zero padding
        bboxes *= max_line
        # map to the image without zero padding
        bboxes -= (offset * max_line)

        CLASSES = VOC_CLASSES
        class_color = tools.CLASS_COLOR
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img_raw, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color[int(cls_indx)], 2)
                cv2.rectangle(img_raw, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_color[int(cls_indx)], -1)
                mess = '%s' % (CLASSES[int(cls_indx)])
                cv2.putText(img_raw, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.imshow('detection', img_raw)
        cv2.waitKey(0)
        # print('Saving the' + str(index) + '-th image ...')
        # cv2.imwrite('test_images/' + args.dataset+ '3/' + str(index).zfill(6) +'.jpg', img)



def test():
    # get device
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load net
    num_classes = len(VOC_CLASSES)
    input_size = [args.input_size, args.input_size]
    testset = VOCDetection(args.voc_root, [('2007', 'test')], BaseTransform(input_size))

    if args.version == 'yolo_v3_spp':
        from models.yolo_v3_spp import YOLOv3SPP
        net = YOLOv3SPP(device, input_size=input_size, num_classes=num_classes, anchor_size=config.MULTI_ANCHOR_SIZE)
    
    elif args.version == 'tiny_yolo_v3_spp':
        from models.tiny_yolo_v3_spp import YOLOv3SPPtiny
    
        net = YOLOv3SPPtiny(device, input_size=input_size, num_classes=num_classes, anchor_size=config.TINY_MULTI_ANCHOR_SIZE)
        print('Let us test tiny-yolo-v3-spp on the VOC0712 dataset ......')

    net.load_state_dict(torch.load(args.trained_model, map_location=device))
    net.to(device).eval()
    print('Finished loading model!')

    # evaluation
    test_net(net, device, testset, input_size,
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test()