import os
import csv
import json
import logging
import random
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

from smoke.modeling.heatmap_coder import (
    get_transfrom_matrix,
    affine_transform,
    gaussian_radius,
    draw_umich_gaussian,
)
from smoke.modeling.smoke_coder import encode_label
from smoke.structures.params_3d import ParamsList

TYPE_ID_CONVERSION = {
    'bicycle': 0,
    'bus': 1,
    'car': 2,
    'construction_vehicle': 3,
    'motorcycle': 4,
    'pedestrian': 5,
    'trailer': 6,
    'truck': 7
}


class NuScenesDataset(Dataset):
    def __init__(self, cfg, root, json_file, is_train=True, transforms=None):
        super(NuScenesDataset, self).__init__()
        self.root = root
        self.json_file = os.path.join(root, json_file)
        with open(self.json_file, 'r') as f:
            infos = json.load(f)
        self.image_infos = infos['images']
        self.anns_infos = infos['annotations']

        self.is_train = is_train
        self.transforms = transforms

        self.classes = cfg.DATASETS.DETECT_CLASSES
        if self.is_train:
            self.filter_samples(self.classes)
        self.num_samples = len(self.image_infos)

        self.flip_prob = cfg.INPUT.FLIP_PROB_TRAIN if is_train else 0
        self.aug_prob = cfg.INPUT.SHIFT_SCALE_PROB_TRAIN if is_train else 0
        self.shift_scale = cfg.INPUT.SHIFT_SCALE_TRAIN
        self.num_classes = len(self.classes)

        self.input_width = cfg.INPUT.WIDTH_TRAIN
        self.input_height = cfg.INPUT.HEIGHT_TRAIN
        self.output_width = self.input_width // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_height = self.input_height // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.max_objs = cfg.DATASETS.MAX_OBJECTS

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing NuScenes {} with {} samples loaded".format(json_file, self.num_samples))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # load default parameter here
        img, image_token, anns, K = self.load_data(idx)

        center = np.array([i / 2 for i in img.size], dtype=np.float32)
        size = np.array([i for i in img.size], dtype=np.float32)

        """
        resize, horizontal flip, and affine augmentation are performed here.
        since it is complicated to compute heatmap w.r.t transform.
        """
        flipped = False
        if (self.is_train) and (random.random() < self.flip_prob):
            flipped = True
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            center[0] = size[0] - center[0] - 1
            K[0, 2] = size[0] - K[0, 2] - 1
            K[0, 3] *= -1

        affine = False
        if (self.is_train) and (random.random() < self.aug_prob):
            affine = True
            shift, scale = self.shift_scale[0], self.shift_scale[1]
            shift_ranges = np.arange(-shift, shift + 0.1, 0.1)
            center[0] += size[0] * random.choice(shift_ranges)
            center[1] += size[1] * random.choice(shift_ranges)

            scale_ranges = np.arange(1 - scale, 1 + scale + 0.1, 0.1)
            size *= random.choice(scale_ranges)

        center_size = [center, size]
        trans_affine = get_transfrom_matrix(
            center_size,
            [self.input_width, self.input_height]
        )
        trans_affine_inv = np.linalg.inv(trans_affine)
        img = img.transform(
            (self.input_width, self.input_height),
            method=Image.AFFINE,
            data=trans_affine_inv.flatten()[:6],
            resample=Image.BILINEAR,
        )

        trans_mat = get_transfrom_matrix(
            center_size,
            [self.output_width, self.output_height]
        )

        if not self.is_train:
            # for inference we parametrize with original size
            target = ParamsList(image_size=size,
                                is_train=self.is_train)
            target.add_field("trans_mat", trans_mat)
            target.add_field("K", K)
            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target, image_token

        heat_map = np.zeros([self.num_classes, self.output_height, self.output_width], dtype=np.float32)
        regression = np.zeros([self.max_objs, 3, 8], dtype=np.float32)
        cls_ids = np.zeros([self.max_objs], dtype=np.int32)
        proj_points = np.zeros([self.max_objs, 2], dtype=np.int32)
        p_offsets = np.zeros([self.max_objs, 2], dtype=np.float32)
        dimensions = np.zeros([self.max_objs, 3], dtype=np.float32)
        locations = np.zeros([self.max_objs, 3], dtype=np.float32)
        rotys = np.zeros([self.max_objs], dtype=np.float32)
        reg_mask = np.zeros([self.max_objs], dtype=np.uint8)
        flip_mask = np.zeros([self.max_objs], dtype=np.uint8)

        for i, a in enumerate(anns):
            a = a.copy()
            cls = a["label"]

            locs = np.array(a["locations"])
            rot_y = np.array(a["rot_y"])
            if flipped:
                locs[0] *= -1
                rot_y *= -1

            point, box2d, box3d = encode_label(
                K, rot_y, a["dimensions"], locs
            )
            point = affine_transform(point, trans_mat)
            box2d[:2] = affine_transform(box2d[:2], trans_mat)
            box2d[2:] = affine_transform(box2d[2:], trans_mat)
            box2d[[0, 2]] = box2d[[0, 2]].clip(0, self.output_width - 1)
            box2d[[1, 3]] = box2d[[1, 3]].clip(0, self.output_height - 1)
            h, w = box2d[3] - box2d[1], box2d[2] - box2d[0]

            if (0 < point[0] < self.output_width) and (0 < point[1] < self.output_height):
                point_int = point.astype(np.int32)
                p_offset = point - point_int
                radius = gaussian_radius(h, w)
                radius = max(0, int(radius))
                heat_map[cls] = draw_umich_gaussian(heat_map[cls], point_int, radius)

                cls_ids[i] = cls
                regression[i] = box3d
                proj_points[i] = point_int
                p_offsets[i] = p_offset
                dimensions[i] = np.array(a["dimensions"])
                locations[i] = locs
                rotys[i] = rot_y
                reg_mask[i] = 1 if not affine else 0
                flip_mask[i] = 1 if not affine and flipped else 0

        target = ParamsList(image_size=img.size,
                            is_train=self.is_train)
        target.add_field("hm", heat_map)
        target.add_field("reg", regression)
        target.add_field("cls_ids", cls_ids)
        target.add_field("proj_p", proj_points)
        target.add_field("dimensions", dimensions)
        target.add_field("locations", locations)
        target.add_field("rotys", rotys)
        target.add_field("trans_mat", trans_mat)
        target.add_field("K", K)
        target.add_field("reg_mask", reg_mask)
        target.add_field("flip_mask", flip_mask)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, image_token

    def load_data(self, idx):
        image_info = self.image_infos[idx]
        img_path = os.path.join(self.root, image_info['filename'])
        img = Image.open(img_path)
        image_token = image_info['token']
        K = np.array(image_info['cam_intrinsic'], dtype=np.float32)
        
        anns_info = self.anns_infos[idx]
        annotations = []

        if self.is_train:
            for ann_info in anns_info:
                annotations.append({
                    "class": ann_info['det_name'],
                    "label": TYPE_ID_CONVERSION[ann_info['det_name']],
                    "dimensions": [float(ann_info['wlh'][1]), float(ann_info['wlh'][2]), float(ann_info['wlh'][0])],
                    "locations": [float(ann_info['location'][0]), float(ann_info['location'][1]), float(ann_info['location'][2])],
                    "rot_y": float(ann_info['rot_y'])
                })

        return img, image_token, annotations, K
    
    def filter_samples(self, classes):
        image_infos_filtered = []
        anns_infos_filtered = []
        for idx in range(len(self.image_infos)):
            anns_info_filtered = []
            for ann_info in self.anns_infos[idx]:
                if ann_info['det_name'] in classes:
                    anns_info_filtered.append(ann_info)
            if anns_info_filtered:
                image_infos_filtered.append(self.image_infos[idx])
                anns_infos_filtered.append(anns_info_filtered)
        
        self.image_infos = image_infos_filtered
        self.anns_infos = anns_infos_filtered
        return

# debug
if __name__ == "__main__":
    from smoke.config import cfg_nusc as cfg
    root = '../../../datasets/nuscenes/'
    json_file = 'smoke_convert/train_half.json'
    train_full = NuScenesDataset(cfg, root, json_file)
    print(len(train_full))