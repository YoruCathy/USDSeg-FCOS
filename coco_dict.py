import pickle
from sklearn.decomposition import DictionaryLearning
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import os
from PIL import Image
from time import time
from sklearn.metrics import jaccard_score
import sys

import warnings
from tqdm import tqdm


class Config:
    def __init__(self):
        self.dataset = 'coco'
        self.path = '/home/tutian/dataset/coco/bases_64'
        self.scale = (64, 64)
        self.n_components = 32
        self.n_iter = [1, 10, 50, 200]
        self.dict = self.get_cat_dict(self.dataset)
        self.save = True  # Whether to save the model
        # The relative path of trained model
        self.save_path = f'sparse_{self.dataset}_{self.scale[0]}_{self.scale[1]}'
        # Create the folder
        if self.save:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
        self.allow_save = True  # Whether to allow load saved model

        self.select_cat = None
        if self.select_cat is None:
            self.select_cat = self.dict.keys()

        if type(self.n_components) == type(1):
            self.n_components = [self.n_components]
        if type(self.n_iter) == type(1):
            self.n_iter = [self.n_iter]

    def get_cat_dict(self, dataset):
        coco_dict = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
                     9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                     16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
                     24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
                     34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                     40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
                     46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
                     53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
                     60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                     70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
                     78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
                     86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

        sbd_dict = {1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat",
                    9: "chair", 10: "cow", 11: "dining table", 12: "dog", 13: "horse", 14: "motorbike", 15: "person",
                    16: "potted plant", 17: "sheep", 18: "sofa", 19: "train", 20: "monitor"}
        if(dataset == 'coco'):
            return coco_dict
        elif (dataset == 'sbd'):
            return sbd_dict
        else:
            raise Exception('dataset must be either coco or sbd')


class Logger:
    def __init__(self):
        self.storage = []

    def add(self, id, n_comp, n_iter, mIOU, extra_info=''):
        self.storage.append({'id': id, 'n_comp': n_comp,
                             'n_iter': n_iter, 'mIOU': mIOU, 'best_IOU': extra_info})

    def print(self):
        print('|'.join(list(self.storage[0].keys())))
        for info in self.storage:
            first = True
            for _, value in info.items():
                if first:
                    print(f'{value}', end='')
                    first = False
                else:
                    print(f'|{value}', end='')
            print('\n', end='')

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write('|'.join(list(self.storage[0].keys())) + '\n')
            for info in self.storage:
                first = True
                for _, value in info.items():
                    if first:
                        f.write(f'{value}')
                        first = False
                    else:
                        f.write(f'|{value}')
                f.write('\n')


class Timer:
    def __init__(self):
        self.time = time()

    def stop(self, print_=True, refresh=True, start='', end='\n'):
        delta_time = time() - self.time
        if print_:
            print(start + "%.2f seconds used" %
                  (time() - self.time), end=end)
        if refresh:
            self.time = time()
        return delta_time


def readCategory(path, cat_dict, cat_id, scale, normalize):
    path += "_" + str(scale[0])
    mask_list = os.listdir(path + "/" + cat_dict[cat_id])
    all_masks = np.zeros((len(mask_list), scale[0] * scale[1]))
    for i in tqdm(range(len(mask_list))):
        mask = np.array(Image.open(
            path + "/" + cat_dict[cat_id] + "/" + mask_list[i]))
        all_masks[i] = mask.flatten()
    if normalize:
        all_masks = (all_masks - all_masks.mean(axis=0)) / \
            all_masks.std(axis=0)
    return all_masks


def readAllMasks(path, scale, normalize):
    tmp = Timer()
    path += "_" + str(scale[0])
    # mask_list = os.listdir(path + "/" + cat_dict[cat_id])

    mask_list = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            mask_list.append(os.path.join(r, file))

    all_masks = np.zeros((len(mask_list), scale[0] * scale[1]))
    for i, img in enumerate(tqdm(mask_list)):
        mask = np.array(Image.open(img))
        all_masks[i] = mask.flatten()
    if normalize:
        all_masks = (all_masks - all_masks.mean(axis=0)) / \
            all_masks.std(axis=0)

    tmp.stop()
    return all_masks


def scale_to_255(bases):
    new = (bases - bases.min()) / (bases.max() - bases.min()) * 255
    return new.astype('uint8')


def binarize(patches, threshold=128):
    return (patches > threshold) * 255


def visualizeTwoImageWithIOU(recon, origin):
    iou = jaccard_score(origin, recon)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(recon.reshape(cfg.scale))
    axes[0].set_title("recon:" + str(iou))
    axes[1].imshow(origin.reshape(cfg.scale))
    axes[1].set_title("origin")
    plt.show()


def check_version():
    import sklearn
    if sklearn.__version__ == '0.21.3':
        print(sklearn.__version__,  'checked')
    else:
        print('[Warning] You\'r using sklearn', sklearn.__version__)


if __name__ == '__main__':
    cfg = Config()
    log = Logger()
    timer = Timer()

    check_version()

    images = readAllMasks(cfg.path, cfg.scale, False)  # Train images
    print(f"train images {images.shape[0]}")
    timer.stop(print_=False)

    for n_iter in cfg.n_iter:
        for n_components in cfg.n_components:
            print(
                f'Start training dictionary with {n_components} bases and {n_iter} max iters')

            hit = False
            if cfg.allow_save and os.path.exists(f'{cfg.save_path}/all_{n_components}_{n_iter}.sklearnmodel'):
                # Use pretrained model
                dico = pickle.load(
                    open(f'{cfg.save_path}/all_{n_components}_{n_iter}.sklearnmodel', 'rb'))
                print(
                    f'Use hitted {cfg.save_path}/all_{n_components}_{n_iter}.sklearnmodel')
                hit = True
            else:
                # Train a new model
                dico = DictionaryLearning(
                    n_components=n_components, n_jobs=-24, max_iter=n_iter, verbose=True)
                dico.fit(images)
                n_iter_actual = dico.n_iter_
                print(f'{n_iter_actual} iters')

            timer.stop(start=' ')

            # Save the model
            if cfg.save:
                np.save(
                    f'{cfg.save_path}/all_{n_components}_{n_iter_actual}', dico.components_)
                pickle.dump(dico, open(
                    f'{cfg.save_path}/all_{n_components}_{n_iter_actual}.sklearnmodel', 'wb'))

    log.print()
