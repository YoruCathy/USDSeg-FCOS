from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
config_file = 'configs/usdseg/usd_r50_caffe_fpn_gn_1x_4gpu.py'
checkpoint_file = 'work_dirs/fcos_r50_usd/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# # test a single image and show the results
# img = 'path-to-img'
# or img = mmcv.imread(img), which will only load it once
# result = inference_detector(model, img)
# # or save the visualization results to image files
# show_result(img, result, model.CLASSES, out_file='result.jpg')

for image in os.listdir('data/coco/val2017'):
    img = mmcv.imread('data/coco/val2017/'+image)
    # print(img)
    result = inference_detector(model, img)
    show_result(img, result, model.CLASSES,
                out_file='work_dirs/vis/2/'+image, show=False)
