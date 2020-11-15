import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data.cocodataset import *
from data import config, BaseTransform, VOCAnnotationTransform, VOCDetection, VOC_ROOT, VOC_CLASSES
from utils import get_device
import numpy as np
import cv2
import time
from decimal import *


parser = argparse.ArgumentParser(description='YOLO-v2 Detection')
parser.add_argument('-v', '--version', default='yolo_v3_spp',
                    help='yolo_v3_spp, tiny_yolo_v3_spp')
parser.add_argument('-d', '--dataset', default='COCO',
                    help='we use VOC-test or COCO-val to test.')
parser.add_argument('--trained_model', default='weights/yolo_v2_72.2.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--visual_threshold', default=0.3, type=float,
                    help='Final confidence threshold')
parser.add_argument('--dataset_root', default='/home/k545/object-detection/dataset/COCO/', 
                    help='Location of COCO root directory')
parser.add_argument('--cuda', action='store_true', default=False, 
                    help='use cuda.')
parser.add_argument('-f', default=None, type=str, 
                    help="Dummy arg so we can load in Jupyter Notebooks")
parser.add_argument('--debug', action='store_true', default=False,
                    help='debug mode where only one image is trained')


args = parser.parse_args()

coco_class_labels = ('background',
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                        'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

coco_class_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                    70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

def test_net(net, device, testset, thresh, mode='voc'):
    class_color = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(80)]
    num_images = len(testset)

    path_save = os.path.join('test_images/', args.dataset, args.version) 
    os.makedirs(path_save, exist_ok=True)

    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        if args.dataset == 'COCO':
            img, _ = testset.pull_image(index)
            img_tensor, _, h, w, offset, scale = testset.pull_item(index)
        elif args.dataset == 'VOC':
            img = testset.pull_image(index)
            img_tensor, _, h, w, offset, scale = testset.pull_item(index)

        x = img_tensor.unsqueeze(0).to(device)

        t0 = time.clock()
        bboxes, scores, cls_inds = net(img_tensor)
        print("detection time used ", Decimal(time.clock()) - Decimal(t0), "s")
        # scale each detection back up to the image
        max_line = max(h, w)
        # map the boxes to input image with zero padding
        bboxes *= max_line
        # map to the image without zero padding
        bboxes -= (offset * max_line)

        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color[int(cls_indx)], 1)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_color[int(cls_indx)], -1)
                cls_id = coco_class_index[int(cls_indx)]
                cls_name = coco_class_labels[cls_id]
                # mess = '%s: %.3f' % (cls_name, scores[i])
                mess = '%s' % (cls_name)
                cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.imshow('detection', img)
        cv2.waitKey(0)
        # if index % 500 == 0:
        #     print('Saving ' + str(index) + '-th image ...')
        # cv2.imwrite(os.path.join(path_save, str(index).zfill(6) +'.jpg'), img)



def test():
    # get device
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load net
    num_classes = 80
    if args.dataset == 'COCO':
        cfg = config.coco_ab
        testset = COCODataset(
                    data_dir=args.dataset_root,
                    json_file='instances_val2017.json',
                    name='val2017',
                    img_size=cfg['min_dim'][0],
                    transform=BaseTransform(cfg['min_dim']),
                    debug=args.debug)
    elif args.dataset == 'VOC':
        cfg = config.voc_ab
        testset = VOCDetection(VOC_ROOT, [('2007', 'test')], BaseTransform(cfg['min_dim']))
    

    if args.version == 'yolo_v3_spp':
        from models.yolo_v3_spp import YOLOv3SPP
        net = YOLOv3SPP(device, input_size=cfg['min_dim'], num_classes=num_classes, anchor_size=config.MULTI_ANCHOR_SIZE_COCO)
   
    elif args.version == 'tiny_yolo_v3_spp':
        from models.tiny_yolo_v3_spp import YOLOv3SPPtiny
    
        net = YOLOv3SPPtiny(device, input_size=cfg['min_dim'], num_classes=num_classes, anchor_size=config.TINY_MULTI_ANCHOR_SIZE_COCO)

    net.load_state_dict(torch.load(args.trained_model, map_location='cuda'))
    net.to(device).eval()
    print('Finished loading model!')

    # evaluation
    test_net(net, device, testset, thresh=args.visual_threshold)

if __name__ == '__main__':
    test()