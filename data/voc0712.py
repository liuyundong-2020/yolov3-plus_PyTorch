"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# note: if you used our download scripts, this should be right
VOC_ROOT = "/home/k545/object-detection/dataset/VOCdevkit/"

class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712', mosaic=False):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        self.mosaic = mosaic
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w, offset, scale = self.pull_item(index)

        return im, gt

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

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        # mosaic augmentation
        if self.mosaic and np.random.randint(2):
            ids_list_ = self.ids[:index] + self.ids[index+1:]
            # random sample 3 indexs
            id2, id3, id4 = random.sample(ids_list_, 3)
            ids = [id2, id3, id4]
            img_lists = [img]
            tg_lists = [target]
            for id_ in ids:
                img_ = cv2.imread(self._imgpath % id_)
                height_, width_, channels_ = img_.shape

                target_ = ET.parse(self._annopath % id_).getroot()              
                target_ = self.target_transform(target_, width_, height_)

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

            return torch.from_numpy(mosaic_img).permute(2, 0, 1), target, height, width, offset, scale

        # basic augmentation(SSDAugmentation or BaseTransform)
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
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


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

    # dataset
    dataset = VOCDetection(VOC_ROOT, [('2007', 'trainval')],
                            BaseTransform([416, 416], (0, 0, 0)),
                            VOCAnnotationTransform(), mosaic=True)
    for i in range(1000):
        im, gt, h, w, _, _ = dataset.pull_item(i)
        img = im.permute(1,2,0).numpy()[:, :, (2, 1, 0)].astype(np.uint8)
        for box in gt:
            xmin, ymin, xmax, ymax, _ = box
            xmin *= 416
            ymin *= 416
            xmax *= 416
            ymax *= 416
            img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255), 2)
        cv2.imshow('gt', img)
        cv2.waitKey(0)
