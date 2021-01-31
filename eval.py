import torch
import torch.nn as nn
from data import *
import argparse
from utils.vocapi_evaluator import VOCAPIEvaluator
from utils.cocoapi_evaluator import COCOAPIEvaluator


parser = argparse.ArgumentParser(description='YOLO-v2 Detector Evaluation')
parser.add_argument('-v', '--version', default='yolo_v3_plus',
                    help='yolo_v3_plus, yolo_v3_plus_x, yolo_v3_plus_large, yolo_v3_plus_medium, yolo_v3_plus_small, \
                            yolo_v3_slim, yolo_v3_slim_csp.')
parser.add_argument('-d', '--dataset', default='voc',
                    help='voc, coco-val, coco-test.')
parser.add_argument('--trained_model', type=str,
                    default='weights_yolo_v2/yolo_v2_72.2.pth', 
                    help='Trained state_dict file path to open')
parser.add_argument('-size', '--input_size', default=416, type=int,
                    help='input_size')
parser.add_argument('-ct', '--conf_thresh', default=0.001, type=float,
                    help='conf thresh')
parser.add_argument('-nt', '--nms_thresh', default=0.50, type=float,
                    help='nms thresh')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Use cuda')
parser.add_argument('--diou_nms', action='store_true', default=False, 
                    help='use diou nms.')

args = parser.parse_args()



def voc_test(model, device, input_size):
    evaluator = VOCAPIEvaluator(data_root=VOC_ROOT,
                                img_size=input_size,
                                device=device,
                                transform=BaseTransform(input_size),
                                labelmap=VOC_CLASSES,
                                display=True
                                )

    # VOC evaluation
    evaluator.evaluate(model)


def coco_test(model, device, input_size, test=False):
    if test:
        # test-dev
        print('test on test-dev 2017')
        evaluator = COCOAPIEvaluator(
                        data_dir=coco_root,
                        img_size=input_size,
                        device=device,
                        testset=True,
                        transform=BaseTransform(input_size)
                        )

    else:
        # eval
        evaluator = COCOAPIEvaluator(
                        data_dir=coco_root,
                        img_size=input_size,
                        device=device,
                        testset=False,
                        transform=BaseTransform(input_size)
                        )

    # COCO evaluation
    evaluator.evaluate(model)


if __name__ == '__main__':
    # dataset
    if args.dataset == 'voc':
        print('eval on voc ...')
        num_classes = 20
        anchor_size = MULTI_ANCHOR_SIZE
    elif args.dataset == 'coco-val':
        print('eval on coco-val ...')
        num_classes = 80
        anchor_size = MULTI_ANCHOR_SIZE_COCO
    elif args.dataset == 'coco-test':
        print('eval on coco-test-dev ...')
        num_classes = 80
        anchor_size = MULTI_ANCHOR_SIZE_COCO
    else:
        print('unknow dataset !! we only support voc, coco-val, coco-test !!!')
        exit(0)

    # cuda
    if args.cuda:
        print('use cuda')
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # input size
    input_size = [args.input_size, args.input_size]

    # build model
    # # yolo_v3_plus series: yolo_v3_plus, yolo_v3_plus_x, yolo_v3_plus_large, yolo_v3_plus_medium, yolo_v3_plus_small
    if args.version == 'yolo_v3_plus':
        from models.yolo_v3_plus import YOLOv3Plus
        backbone = 'd-53'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_plus on the COCO dataset ......')
    
    elif args.version == 'yolo_v3_plus_x':
        from models.yolo_v3_plus import YOLOv3Plus
        backbone = 'csp-x'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_plus_x on the COCO dataset ......')

    elif args.version == 'yolo_v3_plus_large':
        from models.yolo_v3_plus import YOLOv3Plus
        backbone = 'csp-l'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_plus_large on the COCO dataset ......')

    elif args.version == 'yolo_v3_plus_medium':
        from models.yolo_v3_plus import YOLOv3Plus
        backbone = 'csp-m'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_plus_medium on the COCO dataset ......')
    
    elif args.version == 'yolo_v3_plus_small':
        from models.yolo_v3_plus import YOLOv3Plus
        backbone = 'csp-s'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_plus_small on the COCO dataset ......')
    
    # # yolo_v3_slim series: yolo_v3_slim, yolo_v3_slim_csp
    elif args.version == 'yolo_v3_slim':
        from models.yolo_v3_slim import YOLOv3Slim
        backbone = 'd-tiny'
        
        yolo_net = YOLOv3Slim(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_slim on the COCO dataset ......')

    elif args.version == 'yolo_v3_slim_csp':
        from models.yolo_v3_slim import YOLOv3Slim
        backbone = 'csp-slim'
        
        yolo_net = YOLOv3Slim(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_slim_csp on the COCO dataset ......')

    else:
        print('Unknown version !!!')
        exit()

    # load net
    yolo_net.load_state_dict(torch.load(args.trained_model, map_location='cuda'))
    yolo_net.to(device).eval()
    print('Finished loading model!')
    
    # evaluation
    with torch.no_grad():
        if args.dataset == 'voc':
            voc_test(yolo_net, device, input_size)
        elif args.dataset == 'coco-val':
            coco_test(yolo_net, device, input_size, test=False)
        elif args.dataset == 'coco-test':
            coco_test(yolo_net, device, input_size, test=True)
