# config.py

# yolov3-plus with cspdarknet-53
yolov3plus_cfg = {
    # network
    'backbone': 'cd53',
    # for multi-scale trick
    'train_size': 640,
    'val_size': 640,
    'random_size_range': [10, 20],
    # anchor size
    'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                    [30, 61],   [62, 45],   [59, 119],
                    [116, 90],  [156, 198], [373, 326]],
    # train
    'lr_epoch': (100, 150),
    'max_epoch': 200,
    'ignore_thresh': 0.5
}
