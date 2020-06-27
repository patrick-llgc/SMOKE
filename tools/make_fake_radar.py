import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
from smoke.utils import kitti_utils as ku

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--kitti_root', type=str, default='../datasets/kitti', help='KITTI dataset root.')
FLAGS = parser.parse_args()

training_root = os.path.join(FLAGS.kitti_root, 'training')
lidar_root = os.path.join(training_root, 'velodyne')
image_root = os.path.join(training_root, 'image_2')
label_root = os.path.join(training_root, 'label_2')
calib_root = os.path.join(training_root, 'calib')
radar_root = os.path.join(training_root, 'fake_radar')

if os.path.exists(radar_root):
    raise FileExistsError('Fake radar folder exists!')

os.mkdir(radar_root)

with open(os.path.join(training_root, 'ImageSets', 'trainval.txt'), 'r') as f:
    sample_ids = f.read().splitlines()

for sample_id in tqdm(sample_ids):

    # get Lidar point cloud with intensity
    pc_lidar = np.fromfile(os.path.join(lidar_root, sample_id + '.bin'), dtype=np.float32).reshape(-1, 4) # Nx4

    # get calibration
    calib = ku.Calibration(os.path.join(calib_root, sample_id + '.txt'))

    # convert pc_lidar to pc_rect
    pc_rect = calib.lidar_to_rect(pc_lidar[:, :3]) # Nx3
    pc_intensity = pc_lidar[:, 3] # N

    # reduce point cloud to those in camera view
    img = Image.open(os.path.join(image_root, sample_id + '.png'))
    pc_camera_flag = ku.get_pc_in_camera_flag(pc_rect, calib, img.size)
    pc_rect = pc_rect[pc_camera_flag]
    pc_intensity = pc_intensity[pc_camera_flag]

    # get 3D bounding boxes TODO: filter object types
    gt_objects = ku.filter_objects(ku.get_objects_from_file(os.path.join(label_root, sample_id + '.txt')))
    gt_bboxes = ku.objs_to_boxes3d(gt_objects)

    # crop point cloud in 3D bounding boxes
    pc_fg_flag = np.zeros((pc_rect.shape[0]), dtype=np.bool)
    gt_bboxes = ku.box3d_remove_roof(ku.box3d_remove_ground(gt_bboxes))
    gt_corners = ku.boxes3d_to_corners3d(gt_bboxes, rotate=True)
    for k in range(gt_bboxes.shape[0]):
        box_corners = gt_corners[k]
        pc_in_box_flag = ku.in_hull(pc_rect, box_corners)
        pc_fg_flag = np.logical_or(pc_in_box_flag, pc_fg_flag)

    pc_rect = pc_rect[pc_fg_flag]
    pc_intensity = pc_intensity[pc_fg_flag]

    # merge point cloud and intensity, write file
    pc_lidar_rect = np.hstack((pc_rect.astype(np.float32), pc_intensity.reshape(-1, 1).astype(np.float32)))
    pc_lidar_rect.tofile(os.path.join(radar_root, sample_id + '.bin'))