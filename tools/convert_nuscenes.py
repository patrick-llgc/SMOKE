import os

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import train_detect as TRAIN_SCENES_HALF, train as TRAIN_SCENES_FULL, val as VAL_SCENES

DATA_PATH = '../datasets/nuscenes'
OUT_PATH = os.path.join(DATA_PATH, 'smoke_convert')
USED_SENSOR = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
nusc = NuScenes(version='v1.0-trainval', dataroot=DATA_PATH, verbose=True)

SPLITS = {
    'train_half': {'scenes': TRAIN_SCENES_HALF}, 
    'train_full': {'scenes': TRAIN_SCENES_FULL}, 
    'val': {'scenes': VAL_SCENES}
}

for sample in nusc.sample:
    pass