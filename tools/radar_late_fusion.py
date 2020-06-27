import os
import argparse
from tqdm import tqdm
import numpy as np
from smoke.utils import kitti_utils as ku

cos_ = np.cos(np.pi / 4.0)

def bias(theta):
    if np.cos(theta) > cos_ or np.cos(theta - np.pi) > cos_:
        return 1.8
    else:
        return 0.8

def radar_late_fusion(objects, radar_pc):
    objs_frustum = ku.corners3d_to_frustums2d(ku.boxes3d_to_corners3d(ku.objs_to_boxes3d(objects))) # Nx2
    radar_azim = np.arctan2(radar_pc[:, 2], radar_pc[:, 0])
    radar_dist = np.linalg.norm(radar_pc[:, [0, 2]], axis=1)

    for k in range(len(objects)):

        obj = objects[k]
        if obj.cls_type != 'Car':
            continue

        obj_azim = np.arctan2(obj.loc[2], obj.loc[0])
        obj_theta = np.pi - obj.ry - obj_azim
        obj_dist = np.linalg.norm(obj.loc[[0, 2]])
        
        azim_min, azim_max = objs_frustum[k]
        radar_in_frustum_flag = np.logical_and(radar_azim >= azim_min, radar_azim <= azim_max)
        radar_dist_in_frustum = radar_dist[radar_in_frustum_flag]
        radar_scale_factor = (radar_dist_in_frustum + bias(obj_theta)) / obj_dist
        radar_scale_factor_flag = np.logical_and(radar_scale_factor >= 0.7, radar_scale_factor <= 1.3)
        radar_scale_factor = radar_scale_factor[radar_scale_factor_flag]
        if radar_scale_factor.shape[0] == 0:
            continue
        radar_scale_factor = np.min(radar_scale_factor)
        obj.loc[[0, 2]] *= radar_scale_factor

    return objects

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--kitti_root', type=str, default='../datasets/kitti', help='KITTI dataset root.')
parser.add_argument('-p', '--predict_dir', type=str, required=True, help='Folder that contains prediction files')
parser.add_argument('-o', '--output_dir', type=str, default=None, help='Folder to output late fusion predictions')
FLAGS = parser.parse_args()

pred_root = os.path.abspath(FLAGS.predict_dir)
pred_basename = os.path.basename(pred_root)

if 'train' in pred_basename or 'val' in pred_basename:
    subset = 'training'
elif 'test' in pred_basename:
    subset = 'testing'
else:
    raise ValueError('Irregular name of prediction folder!')

subset_root = os.path.join(FLAGS.kitti_root, subset)
radar_root = os.path.join(subset_root, 'fake_radar')

pred_files = os.listdir(pred_root)
sample_ids = []
for pred_file in pred_files:
    if pred_file.endswith('.txt'):
        sample_ids.append(int(pred_file[:-4]))

sample_ids = sorted(sample_ids)
os.mkdir(FLAGS.output_dir)

for sample_id in tqdm(sample_ids):
    objs_pred = ku.get_objects_from_file(os.path.join(pred_root, '%06d.txt'%sample_id))
    radar_pc = np.fromfile(os.path.join(radar_root, '%06d.bin'%sample_id), dtype=np.float32).reshape(-1, 4)
    objs_rect = radar_late_fusion(objs_pred, radar_pc[:, :3])

    objs_rect_str = [obj.to_pred_format() for obj in objs_rect]
    with open(os.path.join(FLAGS.output_dir, '%06d.txt'%sample_id), 'w') as f:
        f.write('\n'.join(objs_rect_str))