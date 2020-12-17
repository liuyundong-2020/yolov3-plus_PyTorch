import os
import numpy as np
import random

import torch
from torch.utils.data import Dataset
import cv2
from pycocotools.coco import COCO


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


class COCODataset(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self, data_dir='COCO', json_file='instances_train2017.json',
                 name='train2017', img_size=416,
                 transform=None, min_size=1, debug=False, mosaic=False):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            min_size (int): bounding boxes smaller than this are ignored
            debug (bool): if True, only one data id is selected from the dataset
        """
        self.data_dir = data_dir
        self.json_file = json_file
        self.coco = COCO(self.data_dir+'annotations/'+self.json_file)
        self.ids = self.coco.getImgIds()
        if debug:
            self.ids = self.ids[1:2]
            print("debug mode...", self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.name = name
        self.max_labels = 50
        self.img_size = img_size
        self.min_size = min_size
        self.transform = transform
        self.mosaic = mosaic

    def __len__(self):
        return len(self.ids)

    def preprocess(self, img, target, height, width):
        # zero padding
        if height > width:
            img_ = np.zeros([height, height, 3])
            delta_w = height - width
            left = delta_w // 2
            img_[:, left:left+width, :] = img
            offset = np.array([[ left / height, 0.,  left / height, 0.]])
            scale =  np.array([[width / height, 1., width / height, 1.]])

        elif height < width:
            img_ = np.zeros([width, width, 3])
            delta_h = width - height
            top = delta_h // 2
            img_[top:top+height, :, :] = img
            offset = np.array([[0.,    top / width, 0.,    top / width]])
            scale =  np.array([[1., height / width, 1., height / width]])
        
        else:
            img_ = img
            scale =  np.array([[1., 1., 1., 1.]])
            offset = np.zeros([1, 4])

        return img_, scale, offset

    def pull_image(self, index):
        id_ = self.ids[index]
        img_file = os.path.join(self.data_dir, self.name,
                                '{:012}'.format(id_) + '.jpg')
        img = cv2.imread(img_file)

        if self.json_file == 'instances_val5k.json' and img is None:
            img_file = os.path.join(self.data_dir, 'train2017',
                                    '{:012}'.format(id_) + '.jpg')
            img = cv2.imread(img_file)

        elif self.json_file == 'image_info_test-dev2017.json' and img is None:
            img_file = os.path.join(self.data_dir, 'test2017',
                                    '{:012}'.format(id_) + '.jpg')
            img = cv2.imread(img_file)

        elif self.json_file == 'image_info_test2017.json' and img is None:
            img_file = os.path.join(self.data_dir, 'test2017',
                                    '{:012}'.format(id_) + '.jpg')
            img = cv2.imread(img_file)

        return img, id_

    def pull_anno(self, index):
        id_ = self.ids[index]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)
        
        target = []
        for anno in annotations:
            if 'bbox' in anno:
                xmin = np.max((0, anno['bbox'][0]))
                ymin = np.max((0, anno['bbox'][1]))
                xmax = xmin + anno['bbox'][2]
                ymax = ymin + anno['bbox'][3]
                
                if anno['area'] > 0 and xmax >= xmin and ymax >= ymin:
                    label_ind = anno['category_id']
                    cls_id = self.class_ids.index(label_ind)

                    target.append([xmin, ymin, xmax, ymax, cls_id])  # [xmin, ymin, xmax, ymax, label_ind]
            else:
                print('No bbox !!')
        return target

    def __getitem__(self, index):
        im, gt, h, w, offset, scale = self.pull_item(index)

        return im, gt

    def pull_item(self, index):
        id_ = self.ids[index]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)

        # load image and preprocess
        img_file = os.path.join(self.data_dir, self.name,
                                '{:012}'.format(id_) + '.jpg')
        img = cv2.imread(img_file)
        
        if self.json_file == 'instances_val5k.json' and img is None:
            img_file = os.path.join(self.data_dir, 'train2017',
                                    '{:012}'.format(id_) + '.jpg')
            img = cv2.imread(img_file)

        assert img is not None

        height, width, channels = img.shape
        
        # COCOAnnotation Transform
        # start here :
        target = []
        for anno in annotations:
            x1 = np.max((0, anno['bbox'][0]))
            y1 = np.max((0, anno['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, anno['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, anno['bbox'][3] - 1))))
            if anno['area'] > 0 and x2 >= x1 and y2 >= y1:
                label_ind = anno['category_id']
                cls_id = self.class_ids.index(label_ind)
                x1 /= width
                y1 /= height
                x2 /= width
                y2 /= height

                target.append([x1, y1, x2, y2, cls_id])  # [xmin, ymin, xmax, ymax, label_ind]
        # end here .

        # mosaic augmentation
        if self.mosaic and np.random.randint(2):
            ids_list_ = self.ids[:index] + self.ids[index+1:]
            # random sample 3 indexs
            id2, id3, id4 = random.sample(ids_list_, 3)
            ids = [id2, id3, id4]
            img_lists = [img]
            tg_lists = [target]
            # load other 3 images and targets
            for id_ in ids:
                anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
                annotations = self.coco.loadAnns(anno_ids)

                # load image and preprocess
                img_file = os.path.join(self.data_dir, self.name,
                                        '{:012}'.format(id_) + '.jpg')
                img_ = cv2.imread(img_file)
                
                if self.json_file == 'instances_val5k.json' and img_ is None:
                    img_file = os.path.join(self.data_dir, 'train2017',
                                            '{:012}'.format(id_) + '.jpg')
                    img_ = cv2.imread(img_file)

                assert img_ is not None

                height_, width_, channels_ = img_.shape             
                # COCOAnnotation Transform
                # start here :
                target_ = []
                for anno in annotations:
                    x1 = np.max((0, anno['bbox'][0]))
                    y1 = np.max((0, anno['bbox'][1]))
                    x2 = np.min((width_ - 1, x1 + np.max((0, anno['bbox'][2] - 1))))
                    y2 = np.min((height_ - 1, y1 + np.max((0, anno['bbox'][3] - 1))))
                    if anno['area'] > 0 and x2 >= x1 and y2 >= y1:
                        label_ind = anno['category_id']
                        cls_id = self.class_ids.index(label_ind)
                        x1 /= width_
                        y1 /= height_
                        x2 /= width_
                        y2 /= height_

                        target_.append([x1, y1, x2, y2, cls_id])  # [xmin, ymin, xmax, ymax, label_ind]
                # end here .
                img_lists.append(img_)
                tg_lists.append(target_)
            # preprocess
            img_processed_lists = []
            tg_processed_lists = []
            for img, target in zip(img_lists, tg_lists):
                h, w, _ = img.shape
                img_, scale, offset = self.preprocess(img, target, h, w)
                if len(target) == 0:
                    target = np.zeros([1, 5])
                else:
                    target = np.array(target)
                    target[:, :4] = target[:, :4] * scale + offset
                # augmentation
                img, boxes, labels = self.transform(img_, target[:, :4], target[:, 4])
                # to rgb
                img = img[:, :, (2, 1, 0)]
                # img = img.transpose(2, 0, 1)
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
                img_processed_lists.append(img)
                tg_processed_lists.append(target)
            # Then, we use mosaic augmentation
            img_size = self.transform.size[0]
            mosaic_img = np.zeros([img_size*2, img_size*2, 3])
            
            img_1, img_2, img_3, img_4 = img_processed_lists
            tg_1, tg_2, tg_3, tg_4 = tg_processed_lists
            # stitch images
            mosaic_img[:img_size, :img_size] = img_1
            mosaic_img[:img_size, img_size:] = img_2
            mosaic_img[img_size:, :img_size] = img_3
            mosaic_img[img_size:, img_size:] = img_4
            mosaic_img = cv2.resize(mosaic_img, (img_size, img_size))
            # modify targets
            tg_1[:, :4] /= 2.0
            tg_2[:, :4] = (tg_2[:, :4] + np.array([1., 0., 1., 0.])) / 2.0
            tg_3[:, :4] = (tg_3[:, :4] + np.array([0., 1., 0., 1.])) / 2.0
            tg_4[:, :4] = (tg_4[:, :4] + 1.0) / 2.0
            target = np.concatenate([tg_1, tg_2, tg_3, tg_4], axis=0)

            return torch.from_numpy(mosaic_img).permute(2, 0, 1).float(), target, height, width, offset, scale

        if self.transform is not None:
            # preprocess
            img_, scale, offset = self.preprocess(img, target, height, width)

            if len(target) == 0:
                target = np.zeros([1, 5])
            else:
                target = np.array(target)
                target[:, :4] = target[:, :4] * scale + offset

            img, boxes, labels = self.transform(img_, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

            return torch.from_numpy(img).permute(2, 0, 1), target, height, width, offset, scale


if __name__ == "__main__":
    def base_transform(image, size, mean):
        x = cv2.resize(image, (size[1], size[0])).astype(np.float32)
        x -= mean
        x = x.astype(np.float32)
        return x

    class BaseTransform:
        def __init__(self, size, mean):
            self.size = size
            self.mean = np.array(mean, dtype=np.float32)

        def __call__(self, image, boxes=None, labels=None):
            return base_transform(image, self.size, self.mean), boxes, labels

    img_size = 640
    dataset = COCODataset(
                data_dir='/home/k545/object-detection/dataset/COCO/',
                img_size=img_size,
                transform=BaseTransform([img_size, img_size], (0, 0, 0)),
                debug=False,
                mosaic=True)
    
    for i in range(1000):
        im, gt, h, w, offset, scale = dataset.pull_item(i)
        img = im.permute(1,2,0).numpy()[:, :, (2, 1, 0)].astype(np.uint8)
        for box in gt:
            xmin, ymin, xmax, ymax, _ = box
            xmin *= img_size
            ymin *= img_size
            xmax *= img_size
            ymax *= img_size
            img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255), 1)
        cv2.imshow('gt', img)
        cv2.waitKey(0)
