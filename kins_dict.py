import os
import pycocotools
import cvbase as cvb
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
import matplotlib.pyplot as plt
import pylab
import skimage.io as io
import numpy as np
import cv2
import cvbase as cvb
import os
import pycocotools.mask as maskUtils
import pdb
import pickle
from sklearn.decomposition import DictionaryLearning
from tqdm import tqdm
from PIL import Image

def make_json_dict(imgs, anns):
	imgs_dict = {}
	anns_dict = {}
	for ann in anns:
		image_id = ann["image_id"]
		if not image_id in anns_dict:
			anns_dict[image_id] = []
			anns_dict[image_id].append(ann)
		else:
			anns_dict[image_id].append(ann)
	
	for img in imgs:
		image_id = img['id']
		imgs_dict[image_id] = img['file_name']

	return imgs_dict, anns_dict

def get_bbox_from_mask(mask):
    coords = np.transpose(np.nonzero(mask))
    xmin = np.min(coords[:, 1])
    xmax = np.max(coords[:, 1])
    ymin = np.min(coords[:, 0])
    ymax = np.max(coords[:, 0])
    return xmin, ymin, xmax, ymax


if __name__ == '__main__':
    mask_list=[]
    src_gt7_path = "/home/ruolin/amodal/data/KINS/annotations/instances_train2017.json"
    anns = cvb.load(src_gt7_path)
    imgs_info = anns['images']
    anns_info = anns["annotations"]

    imgs_dict, anns_dict = make_json_dict(imgs_info, anns_info)
    count = 0
    img = cv2.imread('/home/ruolin/amodal/data/kitti/training/image_2/007480.png', cv2.IMREAD_COLOR)
		
    height, width, _ = img.shape
    for img_id in tqdm(anns_dict.keys()):
        img_name = imgs_dict[img_id]
        anns = anns_dict[img_id]
        if len(anns)==0:
            continue
        for ann in anns:
            amodal_rle = maskUtils.frPyObjects(ann['segmentation'], height, width)
            _amodal_ann_mask = maskUtils.decode(amodal_rle)

            x1, y1, x2, y2 = get_bbox_from_mask(_amodal_ann_mask) 
            t_amodal_ann_mask=_amodal_ann_mask[y1:y2,x1:x2].astype(np.uint8)
            # u=np.unique(t_amodal_ann_mask)
            t_amodal_ann_mask=np.squeeze(t_amodal_ann_mask,-1)
            resized_mask = Image.fromarray(t_amodal_ann_mask).resize((64,64), Image.LINEAR)
            resized_mask = np.reshape(resized_mask, (64*64))
            amodal_ann_mask=2*resized_mask.astype(np.int)-1
            mask_list.append(amodal_ann_mask)
            # print(amodal_ann_mask)

    np.save('/home/ruolin/amodal/data/masks_linear',mask_list)
    # dico=DictionaryLearning(n_components=32,n_jobs=-24,max_iter=20,verbose=2)
    # dico.fit(mask_list)
    # np.save('/home/ruolin/amodal/data/bases', dico.components_)
    # pickle.dump(dico, open('/home/ruolin/amodal/data/bases/amodal.sklearnmodel', 'wb'))