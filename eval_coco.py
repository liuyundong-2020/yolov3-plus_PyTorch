from __future__ import division

from utils.cocoapi_evaluator import COCOAPIEvaluator
from data import *
from utils.augmentations import SSDAugmentation
from data.cocodataset import *
import tools

import os
import random
import argparse
import time
import math
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser(description='YOLO Detection')
parser.add_argument('-v', '--version', default='yolo_v3_spp',
                    help='yolo_v3_spp, yolo_v3_slim')
parser.add_argument('-t', '--testset', action='store_true', default=False,
                    help='COCO_val, COCO_test-dev dataset')
parser.add_argument('-size', '--input_size', default=416, type=int, 
                    help='Batch size for training')
parser.add_argument('--dataset_root', default='/home/k545/object-detection/dataset/COCO/', 
                    help='Location of COCO root directory')
parser.add_argument('--trained_model', default='weights/coco/', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--num_classes', default=80, type=int, 
                    help='The number of dataset classes')
parser.add_argument('--n_cpu', default=8, type=int, 
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Use cuda')
parser.add_argument('--debug', action='store_true', default=False,
                    help='debug mode where only one image is trained')


args = parser.parse_args()
data_dir = args.dataset_root

def test(model, input_size, device):
    if args.testset:
        print('test on test-dev 2017')
        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=input_size,
                        device=device,
                        testset=True,
                        transform=BaseTransform(input_size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229))
                        )

    else:
        # eval
        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=input_size,
                        device=device,
                        testset=False,
                        transform=BaseTransform(input_size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229))
                        )

    # COCO evaluation
    ap50_95, ap50 = evaluator.evaluate(model)
    print('ap50 : ', ap50)
    print('ap50_95 : ', ap50_95)


if __name__ == '__main__':

    input_size = [args.input_size, args.input_size]
    if args.cuda:
        print('use cuda')
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    if args.version == 'yolo_v3_spp':
        from models.yolo_v3_spp import YOLOv3SPP
        model = YOLOv3SPP(device, input_size=input_size, num_classes=args.num_classes, anchor_size=MULTI_ANCHOR_SIZE_COCO)

    elif args.version == 'yolo_v3_plus':
        from models.yolo_v3_plus import YOLOv3Plus
        model = YOLOv3Plus(device, input_size=input_size, num_classes=args.num_classes, anchor_size=MULTI_ANCHOR_SIZE_COCO)

    elif args.version == 'yolo_v3_slim':
        from models.yolo_v3_slim import YOLOv3Slim
    
        model = YOLOv3Slim(device, input_size=input_size, num_classes=args.num_classes, anchor_size=MULTI_ANCHOR_SIZE_COCO)
    
    else:
        print('Unknown Version !!!')
        exit()

    # load model
    model.load_state_dict(torch.load(args.trained_model, map_location=device))
    model.eval().to(device)
    print('Finished loading model!')

    test(model, input_size, device)
