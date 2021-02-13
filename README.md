# yolov3-plus_PyTorch
A better PyTorch version of YOLOv3. 

I call it "YOLOv3Plus"~

It is not the final version. I'm still trying to make it better and better.

# A strong YOLOv3 PyTorch

In this project, you can enjoy several excellent detectors: 

YOLOv3Plus: 
- YOLOv3Plus (with my Darknet53)
- YOLOv3Plus-x (with my CSPDarknet-X)
- YOLOv3Plus-l (with my CSPDarknet-large)
- YOLOv3Plus-m (with my CSPDarknet-medium)
- YOLOv3Plus-s (with my CSPDarknet-small)

YOLOv3Slim:
- YOLOv3Slim (with my Darknet_tiny)
- YOLOv3Slim-csp (with my CSPDarknet-tiny )

Of course, the CSPDarknets used in these new models are all trained by myself on ImageNet. My CSPDarknet is a little different from the one used in YOLOv4 and YOLOv5. I referred to YOLOv4, YOLOv5 and Scaled-YOLOv4. For more details, you can read my backbone files in ```backbone\cspdarknet.py```.

# YOLOv3Plus
I use SPP and PAN module into my YOLOv3Plus, and get a better detector:

On COCO eval:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> size </td><td bgcolor=white> data </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white> AP-S </td><td bgcolor=white> AP-M </td><td bgcolor=white> AP-L </td></tr>


<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Plus</th><td bgcolor=white> 320 </td><td bgcolor=white> COCO eval </td><td bgcolor=white> 34.01 </td><td bgcolor=white> 53.58 </td><td bgcolor=white> 35.1 </td><td bgcolor=white> 12.2 </td><td bgcolor=white> 37.5 </td><td bgcolor=white> 53.8 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Plus</th><td bgcolor=white> 416 </td><td bgcolor=white> COCO eval </td><td bgcolor=white> 37.40 </td><td bgcolor=white> 57.42 </td><td bgcolor=white> 39.0 </td><td bgcolor=white> 17.6 </td><td bgcolor=white> 40.7 </td><td bgcolor=white> 55.9 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Plus</th><td bgcolor=white> 608 </td><td bgcolor=white> COCO eval </td><td bgcolor=white> 40.02 </td><td bgcolor=white> 60.45 </td><td bgcolor=white> 42.3 </td><td bgcolor=white> 22.4 </td><td bgcolor=white> 44.1 </td><td bgcolor=white> 54.2 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8>-</th><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Plus-X</th><td bgcolor=white> 320 </td><td bgcolor=white> COCO eval </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Plus-X</th><td bgcolor=white> 416 </td><td bgcolor=white> COCO eval </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Plus-X</th><td bgcolor=white> 608 </td><td bgcolor=white> COCO eval </td><td bgcolor=white>  </td><td bgcolor=white> </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8>-</th><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Plus-L</th><td bgcolor=white> 320 </td><td bgcolor=white> COCO eval </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Plus-L</th><td bgcolor=white> 416 </td><td bgcolor=white> COCO eval </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Plus-L</th><td bgcolor=white> 608 </td><td bgcolor=white> COCO eval </td><td bgcolor=white>  </td><td bgcolor=white> </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8>-</th><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Plus-M</th><td bgcolor=white> 320 </td><td bgcolor=white> COCO eval </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Plus-M</th><td bgcolor=white> 416 </td><td bgcolor=white> COCO eval </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Plus-M</th><td bgcolor=white> 608 </td><td bgcolor=white> COCO eval </td><td bgcolor=white>  </td><td bgcolor=white> </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8>-</th><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Plus-S</th><td bgcolor=white> 320 </td><td bgcolor=white> COCO eval </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Plus-S</th><td bgcolor=white> 416 </td><td bgcolor=white> COCO eval </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Plus-S</th><td bgcolor=white> 608 </td><td bgcolor=white> COCO eval </td><td bgcolor=white>  </td><td bgcolor=white> </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

</table></tbody>

# YOLOv3Slim
I also provide two lightweight detectors: YOLOv3Slim and YOLOv3Slim-csp.

It is very simple. The backbone network, my darknet_tiny (or my cspdarknet_tiny). The neck is SPP same to as the one used in my YOLOv3Plus. And the head is FPN+PAN with less conv layers and conv kernels.

COCO eval:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> size </td><td bgcolor=white> data </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white> AP-S </td><td bgcolor=white> AP-M </td><td bgcolor=white> AP-L </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Slim</th><td bgcolor=white> 320 </td><td bgcolor=white> COCO eval </td><td bgcolor=white> 23.64 </td><td bgcolor=white> 42.35 </td><td bgcolor=white> 23.0 </td><td bgcolor=white> 6.5 </td><td bgcolor=white> 24.6 </td><td bgcolor=white> 38.6 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Slim</th><td bgcolor=white> 416 </td><td bgcolor=white> COCO eval </td><td bgcolor=white> 26.08 </td><td bgcolor=white> 45.65 </td><td bgcolor=white> 26.2 </td><td bgcolor=white> 9.6 </td><td bgcolor=white> 27.1 </td><td bgcolor=white> 39.7 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Slim</th><td bgcolor=white> 608 </td><td bgcolor=white> COCO eval </td><td bgcolor=white> 26.85 </td><td bgcolor=white> 47.58 </td><td bgcolor=white> 26.7 </td><td bgcolor=white> 13.4 </td><td bgcolor=white> 30.0 </td><td bgcolor=white> 36.0 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8>-</th><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td><td bgcolor=white>-</td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Slim-csp</th><td bgcolor=white> 320 </td><td bgcolor=white> COCO eval </td><td bgcolor=white> 23.19 </td><td bgcolor=white> 41.61 </td><td bgcolor=white> 22.5 </td><td bgcolor=white> 6.0 </td><td bgcolor=white> 23.8 </td><td bgcolor=white> 39.2 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Slim-csp</th><td bgcolor=white> 416 </td><td bgcolor=white> COCO eval </td><td bgcolor=white> 25.67 </td><td bgcolor=white> 45.24 </td><td bgcolor=white> 25.4 </td><td bgcolor=white> 9.5 </td><td bgcolor=white> 26.7 </td><td bgcolor=white> 40.0 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Slim-csp</th><td bgcolor=white> 608 </td><td bgcolor=white> COCO eval </td><td bgcolor=white> 27.21 </td><td bgcolor=white> 47.89 </td><td bgcolor=white> 27.2 </td><td bgcolor=white> 12.3 </td><td bgcolor=white> 30.5 </td><td bgcolor=white> 37.5 </td></tr>
</table></tbody>

I am try to finish this project as soon as possible, but I have no much GPUs to train them. 

Please patiently wait ...


## Installation
- Pytorch-gpu 1.1.0/1.2.0/1.3.0
- Tensorboard 1.14.
- opencv-python, python3.6/3.7

## Dataset

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
python train.py -d voc --cuda -v [select a model] -hr -ms
```

You can run ```python train.py -h``` to check all optional argument.

### COCO
```Shell
python train.py -d coco --cuda -v [select a model] -hr -ms
```


## Test
### VOC
```Shell
python test.py -d voc --cuda -v [select a model] --trained_model [ Please input the path to model dir. ]
```

### COCO
```Shell
python test.py -d coco-val --cuda -v [select a model] --trained_model [ Please input the path to model dir. ]
```


## Evaluation
### VOC
```Shell
python eval.py -d voc --cuda -v [select a model] --train_model [ Please input the path to model dir. ]
```

### COCO
To run on COCO_val:
```Shell
python eval.py -d coco-val --cuda -v [select a model] --train_model [ Please input the path to model dir. ]
```

To run on COCO_test-dev(You must be sure that you have downloaded test2017):
```Shell
python eval.py -d coco-test --cuda -v [select a model] --train_model [ Please input the path to model dir. ]
```
You will get a .json file which can be evaluated on COCO test server.
