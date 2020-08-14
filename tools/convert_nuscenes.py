import os
import copy

import numpy as np
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import mini_train as TRAIN_SCENES_MINI, train_detect as TRAIN_SCENES_HALF, train as TRAIN_SCENES_FULL, mini_val as VAL_SCENES_MINI, val as VAL_SCENES_FULL
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.eval.detection.utils import category_to_detection_name

DEBUG = True
DATA_PATH = '../datasets/nuscenes'
OUT_PATH = os.path.join(DATA_PATH, 'smoke_convert')
USED_CAMS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
nusc = NuScenes(version='v1.0-trainval', dataroot=DATA_PATH, verbose=True)

SPLITS = {
    'train_mini': {'scenes': TRAIN_SCENES_MINI},
    'train_half': {'scenes': TRAIN_SCENES_HALF},
    'train_full': {'scenes': TRAIN_SCENES_FULL},
    'val_mini': {'scenes': VAL_SCENES_MINI},
    'val_full': {'scenes': VAL_SCENES_FULL}
}

for i, sample in enumerate(nusc.sample):
    sample_scene_name = nusc.get('scene', sample['scene_token'])['name']
    splits = []
    for split in SPLITS:
        if sample_scene_name in SPLITS[split]['scenes']:
            splits.append(split)
    if not splits:
        continue

    if DEBUG:
        print(splits)

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

            ann_info = {
                'token': box.token,
                'image_token': image_info['token'],
                'det_name': det_name,
                'location': [box.center[0], box.center[1] + box.wlh[2] / 2.0, box.center[2]], # kitti-format location
                'wlh': box.wlh.tolist(),
                'rot_y': rot_y
            }

        if DEBUG and cam == 'CAM_FRONT':
            print(cam_data)
            print()
            print(cam_calib)
            print()
            print(ego_pose)
            print()
            print(global_T_ego)
            print(global_T_ego.dtype)
            print()
            print(ego_T_cam)
            print(ego_T_cam.dtype)
            print()
            print(global_T_cam)
            print(global_T_cam.dtype)
            print()
            print(boxes)
            print()
            print(cam_intrinsic)
            print(cam_intrinsic.dtype)
            print()
            print(image_info)
            
        



    if DEBUG and i == 0:
        break