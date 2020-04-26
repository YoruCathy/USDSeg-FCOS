from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
from tqdm import tqdm
import sys

config_file = sys.argv[1]
checkpoint_file = sys.argv[2]

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# # test a single image and show the results
# img = 'path-to-img'
# or img = mmcv.imread(img), which will only load it once
# result = inference_detector(model, img)
# # or save the visualization results to image files
# show_result(img, result, model.CLASSES, out_file='result.jpg')
voc_list = ['2007_002227.jpg', '2007_005844.jpg',
            '2007_009258.jpg', '2007_003251.jpg']
coco_list = ['000000011760.jpg', '000000013659.jpg', '000000026564.jpg', '000000032811.jpg', '000000034139.jpg', '000000036660.jpg', '000000050679.jpg', '000000053624.jpg', '000000055167.jpg', '000000058111.jpg', '000000085478.jpg', '000000099114.jpg', '000000103585.jpg', '000000117374.jpg', '000000130699.jpg', '000000131138.jpg',
             '000000136334.jpg', '000000178028.jpg', '000000284445.jpg', '000000328117.jpg', '000000351331.jpg', '000000351362.jpg', '000000357060.jpg', '000000377723.jpg', '000000384666.jpg', '000000414638.jpg', '000000417608.jpg', '000000491008.jpg', '000000526197.jpg', '000000528862.jpg', '000000530146.jpg', '000000549167.jpg', '000000559707.jpg']
for image in tqdm(coco_list, ncols=0):
    img = mmcv.imread('data/coco/val2017/' + image)
    # print(img)
    result = inference_detector(model, img)
    show_result(img, result, model.CLASSES,
                out_file=os.path.join(sys.argv[3], image), show=False)
for image in tqdm(voc_list, ncols=0):
    img = mmcv.imread('/home/ruolin/VOC/VOCdevkit/VOC2012/JPEGImages/' + image)
    # print(img)
    result = inference_detector(model, img)
    show_result(img, result, model.CLASSES,
                out_file=os.path.join(sys.argv[3], image), show=False)
