# yolov3-plus_PyTorch
A better PyTorch version of YOLOv3. 
# A strong YOLOv3 PyTorch

In this project, you can enjoy: 
- yolo-v3-spp
- yolo-v3-plus
- yolo-v3-slim

# YOLOv3-SPP
I try to reproduce YOLOv3 with SPP module.

VOC:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> size </td> <td bgcolor=white> Ours </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC07 test</th> <td bgcolor=white> 416 </td> <td bgcolor=white> 81.6 </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC07 test</th> <td bgcolor=white> 608 </td> <td bgcolor=white> 82.5 </td></tr>
</table></tbody>

COCO:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> data </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white> AP_S </td><td bgcolor=white> AP_M </td><td bgcolor=white> AP_L </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-SPP-320</th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> 31.7 </td><td bgcolor=white> 52.6 </td><td bgcolor=white> 32.9 </td><td bgcolor=white> 10.9 </td><td bgcolor=white> 33.2 </td><td bgcolor=white> 48.6 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-SPP-416</th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> 34.6 </td><td bgcolor=white> 56.1 </td><td bgcolor=white> 36.3 </td><td bgcolor=white> 14.7 </td><td bgcolor=white> 36.2 </td><td bgcolor=white> 50.1 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-SPP-608</th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> 37.1 </td><td bgcolor=white> 58.9 </td><td bgcolor=white> 39.3 </td><td bgcolor=white> 19.6 </td><td bgcolor=white> 39.5 </td><td bgcolor=white> 48.5 </td></tr>
</table></tbody>

So, just have fun !

# YOLOv3-Plus
I add PAN module into the above YOLOv3-SPP, and get a better detector:

On COCO eval:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> data </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-SPP-416</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 37.40 </td><td bgcolor=white> 57.42 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-SPP-608</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 40.02 </td><td bgcolor=white> 60.45 </td></tr>
</table></tbody>


# YOLOv3-Slim
I also provide a lightweight detector: YOLOv3-Slim.

It is very simple. The backbone network, darknet_tiny, consists of only 10 conv layers. The neck is SPP same to as the one used in my YOLOv3-Plus. And the head is FPN+PAN with less conv layers and conv kernels.

COCO eval:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> data </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-Slim-416</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 26.08 </td><td bgcolor=white> 45.65 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-Slim-608</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 26.85 </td><td bgcolor=white> 47.58 </td></tr>
</table></tbody>

</table></tbody>

## Installation
- Pytorch-gpu 1.1.0/1.2.0/1.3.0
- Tensorboard 1.14.
- opencv-python, python3.6/3.7

## Dataset
As for now, I only train and test on PASCAL VOC2007 and 2012. 

### VOC Dataset
I copy the download files from the following excellent project:
https://github.com/amdegroot/ssd.pytorch

I have uploaded the VOC2007 and VOC2012 to BaiDuYunDisk, so for researchers in China, you can download them from BaiDuYunDisk:

Link：https://pan.baidu.com/s/1tYPGCYGyC0wjpC97H-zzMQ 

Password：4la9

You will get a ```VOCdevkit.zip```, then what you need to do is just to unzip it and put it into ```data/```. After that, the whole path to VOC dataset is ```data/VOCdevkit/VOC2007``` and ```data/VOCdevkit/VOC2012```.

#### Download VOC2007 trainval & test

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

#### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

### MSCOCO Dataset
I copy the download files from the following excellent project:
https://github.com/DeNA/PyTorch_YOLOv3

#### Download MSCOCO 2017 dataset
Just run ```sh data/scripts/COCO2017.sh```. You will get COCO train2017, val2017, test2017.


## Train
### VOC
```Shell
python train_voc.py -v [select a model] -hr -ms --cuda
```

You can run ```python train_voc.py -h``` to check all optional argument.

### COCO
```Shell
python train_coco.py -v [select a model] -hr -ms --cuda
```


## Test
### VOC
```Shell
python test_voc.py -v [select a model] --trained_model [ Please input the path to model dir. ] --cuda
```

### COCO
```Shell
python test_coco.py -v [select a model] --trained_model [ Please input the path to model dir. ] --cuda
```


## Evaluation
### VOC
```Shell
python eval_voc.py -v [select a model] --train_model [ Please input the path to model dir. ] --cuda
```

### COCO
To run on COCO_val:
```Shell
python eval_coco.py -v [select a model] --train_model [ Please input the path to model dir. ] --cuda
```

To run on COCO_test-dev(You must be sure that you have downloaded test2017):
```Shell
python eval_coco.py -v [select a model] --train_model [ Please input the path to model dir. ] --cuda -t
```
You will get a .json file which can be evaluated on COCO test server.

You can run ```python train_voc.py -h``` to check all optional argument.

