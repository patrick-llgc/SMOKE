import os
import copy
import json
from tqdm import tqdm

import numpy as np
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import mini_train as TRAIN_SCENES_MINI, train_detect as TRAIN_SCENES_HALF, train as TRAIN_SCENES_FULL, mini_val as VAL_SCENES_MINI, val as VAL_SCENES_FULL
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.kitti import KittiDB
from nuscenes.eval.detection.utils import category_to_detection_name

def _bbox2D_inside(box1, box2): # box1 in box2
    return box1[0] >= box2[0] and box1[2] <= box2[2] and box1[1] >= box2[1] and box1[3] <= box2[3] 

DATA_PATH = '../datasets/nuscenes'
USED_CAMS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
nusc = NuScenes(version='v1.0-trainval', dataroot=DATA_PATH, verbose=True)

SPLITS = {
    'train_mini': {'scenes': TRAIN_SCENES_MINI, 'images': [], 'annotations': []},
    'train_half': {'scenes': TRAIN_SCENES_HALF, 'images': [], 'annotations': []},
    'train_full': {'scenes': TRAIN_SCENES_FULL, 'images': [], 'annotations': []},
    'val_mini': {'scenes': VAL_SCENES_MINI, 'images': [], 'annotations': []},
    'val_full': {'scenes': VAL_SCENES_FULL, 'images': [], 'annotations': []}
}

for sample in tqdm(nusc.sample):
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    splits = []
    for split in SPLITS:
        if scene_name in SPLITS[split]['scenes']:
            splits.append(split)
    if not splits:
        continue

    for cam in USED_CAMS:

        # get data
        cam_token = sample['data'][cam]
        cam_data = nusc.get('sample_data', cam_token)
        cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])

        # compute transformations (4x4 matrices)
        global_T_ego = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)
        ego_T_cam = transform_matrix(cam_calib['translation'], Quaternion(cam_calib['rotation']), inverse=False)
        global_T_cam = np.dot(global_T_ego, ego_T_cam)

        _, boxes, camera_intrinsic = nusc.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
        cam_intrinsic = np.zeros((3, 4))
        cam_intrinsic[:, :3] = camera_intrinsic # 3x3 -> 3x4
        
        # distill image info
        image_info = {
            'token': cam_data['token'],
            'sample_token': cam_data['sample_token'],
            'sensor_name': cam_data['channel'],
            'filename': cam_data['filename'],
            'width': cam_data['width'],
            'height': cam_data['height'],
            'cam_intrinsic': cam_intrinsic.tolist(), # 3x4
            'global_T_ego': global_T_ego.tolist(), # 4x4
            'ego_T_cam': ego_T_cam.tolist(), # 4x4
            'global_T_cam': global_T_cam.tolist(), #4x4
        }

        # distill annotations
        anns_info = []
        for box in boxes:
            det_name = category_to_detection_name(box.name)
            if det_name is None:
                continue

            # estimate rot_y in camera_coordinate
            box_front = box.orientation.rotate(np.array([100.0, 0.0, 0.0]))
            rot_y = -np.arctan2(box_front[2], box_front[0])

            # center to kitti-format location
            box.translate(np.array([0.0, box.wlh[2] / 2.0, 0.0]))

            # compute projected 2D bbox for filtering
            bbox2D = KittiDB.project_kitti_box_to_image(copy.deepcopy(box), cam_intrinsic, imsize=(image_info['width'], image_info['height']))

            ann_info = {
                'token': box.token,
                'image_token': image_info['token'],
                'det_name': det_name,
                'bbox2D': list(bbox2D),
                'location': box.center.tolist(),
                'wlh': box.wlh.tolist(),
                'rot_y': rot_y
            }

            anns_info.append(ann_info)
        
        # a very naive strategy from CenterTrack to filter out object not visible (occluded) in image
        anns_info_filtered = []
        for a in range(len(anns_info)):
            vis = True
            for b in range(len(anns_info)):
                if (anns_info[a]['location'][2] - min(anns_info[a]['wlh']) / 2.0) > (anns_info[b]['location'][2] + max(anns_info[b]['wlh']) / 2.0) and _bbox2D_inside(anns_info[a]['bbox2D'], anns_info[b]['bbox2D']):
                    vis = False
                    break
            if vis:
                anns_info_filtered.append(anns_info[a])

        # save infos to splits
        for split in splits:
            SPLITS[split]['images'].append(copy.deepcopy(image_info))
            SPLITS[split]['annotations'].append(copy.deepcopy(anns_info_filtered))

# dump infos
OUT_PATH = os.path.join(DATA_PATH, 'smoke_convert')
if not os.path.exists(OUT_PATH):
    os.mkdir(OUT_PATH)

for split in SPLITS:
    with open(os.path.join(OUT_PATH, '{}.json'.format(split)), 'w') as f:
        json.dump({'images': SPLITS[split]['images'], 'annotations': SPLITS[split]['annotations']}, f)