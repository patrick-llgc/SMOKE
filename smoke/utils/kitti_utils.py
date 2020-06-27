'''
Copied from PointRCNN
'''
import numpy as np
from scipy.spatial import Delaunay
import scipy
import torch


def get_objects_from_file(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects


def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


def get_pc_in_camera_flag(pts_rect, calib, img_shape):

    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[0])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[1])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    return pts_valid_flag


def filter_objects(objects, black_list=['DontCare', 'Misc']):
    valid_objects = []
    for obj in objects:
        if obj.cls_type not in black_list:
            valid_objects.append(obj)
    return valid_objects


def dist_to_plane(plane, points):
    """
    Calculates the signed distance from a 3D plane to each point in a list of points
    :param plane: (a, b, c, d)
    :param points: (N, 3)
    :return: (N), signed distance of each point to the plane
    """
    a, b, c, d = plane

    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    return (a*x + b*y + c*z + d) / np.sqrt(a**2 + b**2 + c**2)


def rotate_pc_along_y(pc, rot_angle):
    """
    params pc: (N, 3+C), (N, 3) is in the rectified camera coordinate
    params rot_angle: rad scalar
    Output pc: updated pc with XYZ rotated
    """
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


def rotate_pc_along_y_torch(pc, rot_angle):
    """
    :param pc: (N, 512, 3 + C)
    :param rot_angle: (N)
    :return:
    TODO: merge with rotate_pc_along_y_torch in bbox_transform.py
    """
    cosa = torch.cos(rot_angle).view(-1, 1)  # (N, 1)
    sina = torch.sin(rot_angle).view(-1, 1)  # (N, 1)

    raw_1 = torch.cat([cosa, -sina], dim=1)  # (N, 2)
    raw_2 = torch.cat([sina, cosa], dim=1)  # (N, 2)
    R = torch.cat((raw_1.unsqueeze(dim=1), raw_2.unsqueeze(dim=1)), dim=1)  # (N, 2, 2)

    pc_temp = pc[:, :, [0, 2]]  # (N, 512, 2)

    pc[:, :, [0, 2]] = torch.matmul(pc_temp, R.permute(0, 2, 1))  # (N, 512, 2)

    return pc


def boxes3d_to_corners3d(boxes3d, rotate=True):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    :param rotate:
    :return: corners3d: (N, 8, 3)
    """
    boxes_num = boxes3d.shape[0]
    h, w, l = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2.], dtype=np.float32).T  # (N, 8)
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T  # (N, 8)

    y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
    y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)

    if rotate:
        ry = boxes3d[:, 6]
        zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
        rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                             [zeros,       ones,       zeros],
                             [np.sin(ry), zeros,  np.cos(ry)]])  # (3, 3, N)
        R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

        temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), 
                                       y_corners.reshape(-1, 8, 1), 
                                       z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
        rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
        x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)


def boxes3d_to_corners3d_torch(boxes3d, flip=False):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    :return: corners_rotated: (N, 8, 3)
    """
    boxes_num = boxes3d.shape[0]
    h, w, l, ry = boxes3d[:, 3:4], boxes3d[:, 4:5], boxes3d[:, 5:6], boxes3d[:, 6:7]
    if flip:
        ry = ry + np.pi
    centers = boxes3d[:, 0:3]
    zeros = torch.cuda.FloatTensor(boxes_num, 1).fill_(0)
    ones = torch.cuda.FloatTensor(boxes_num, 1).fill_(1)

    x_corners = torch.cat([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2.], dim=1)  # (N, 8)
    y_corners = torch.cat([zeros, zeros, zeros, zeros, -h, -h, -h, -h], dim=1)  # (N, 8)
    z_corners = torch.cat([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dim=1)  # (N, 8)
    corners = torch.cat((x_corners.unsqueeze(dim=1), y_corners.unsqueeze(dim=1), z_corners.unsqueeze(dim=1)), dim=1) # (N, 3, 8)

    cosa, sina = torch.cos(ry), torch.sin(ry)
    raw_1 = torch.cat([cosa, zeros, sina], dim=1)
    raw_2 = torch.cat([zeros, ones, zeros], dim=1)
    raw_3 = torch.cat([-sina, zeros, cosa], dim=1)
    R = torch.cat((raw_1.unsqueeze(dim=1), raw_2.unsqueeze(dim=1), raw_3.unsqueeze(dim=1)), dim=1)  # (N, 3, 3)

    corners_rotated = torch.matmul(R, corners)  # (N, 3, 8)
    corners_rotated = corners_rotated + centers.unsqueeze(dim=2).expand(-1, -1, 8)
    corners_rotated = corners_rotated.permute(0, 2, 1)
    return corners_rotated


def corners3d_to_frustums2d(corners):
    azims = np.arctan2(corners[:, :, 2], corners[:, :, 0])
    azims_min = np.min(azims, axis=1)
    azims_max = np.max(azims, axis=1)
    frustum2d = np.vstack((azims_min, azims_max)).T
    return frustum2d


def boxes3d_to_bev_torch(boxes3d):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    :return:
        boxes_bev: (N, 5) [x1, y1, x2, y2, ry]
    """
    boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))

    cu, cv = boxes3d[:, 0], boxes3d[:, 2]
    half_l, half_w = boxes3d[:, 5] / 2, boxes3d[:, 4] / 2
    boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_l, cv - half_w
    boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_l, cv + half_w
    boxes_bev[:, 4] = boxes3d[:, 6]
    return boxes_bev


def enlarge_box3d(boxes3d, extra_width):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    """
    if isinstance(boxes3d, np.ndarray):
        large_boxes3d = boxes3d.copy()
    else:
        large_boxes3d = boxes3d.clone()
    large_boxes3d[:, 3:6] += extra_width * 2
    large_boxes3d[:, 1] += extra_width
    return large_boxes3d


def box3d_remove_ground(boxes3d, h_prop=0.15):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    """
    if isinstance(boxes3d, np.ndarray):
        rg_boxes3d = boxes3d.copy()
    else:
        rg_boxes3d = boxes3d.clone()
    dh = rg_boxes3d[:, 3] * h_prop
    rg_boxes3d[:, 3] -= dh
    rg_boxes3d[:, 1] -= dh
    return rg_boxes3d

def box3d_remove_roof(boxes3d, h_prop=0.35):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    """
    if isinstance(boxes3d, np.ndarray):
        rf_boxes3d = boxes3d.copy()
    else:
        rf_boxes3d = boxes3d.clone()
    dh = rf_boxes3d[:, 3] * h_prop
    rf_boxes3d[:, 3] -= dh
    return rf_boxes3d

def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag


def objs_to_boxes3d(obj_list):
    boxes3d = np.zeros((obj_list.__len__(), 7), dtype=np.float32)
    for k, obj in enumerate(obj_list):
        boxes3d[k, 0:3], boxes3d[k, 3], boxes3d[k, 4], boxes3d[k, 5], boxes3d[k, 6] \
            = obj.loc, obj.h, obj.w, obj.l, obj.ry
    return boxes3d


def objs_to_scores(obj_list):
    scores = np.zeros((obj_list.__len__()), dtype=np.float32)
    for k, obj in enumerate(obj_list):
        scores[k] = obj.score
    return scores


def get_iou3d(corners3d, query_corners3d, need_bev=False):
    """	
    :param corners3d: (N, 8, 3) in rect coords	
    :param query_corners3d: (M, 8, 3)	
    :return:	
    """
    from shapely.geometry import Polygon
    A, B = corners3d, query_corners3d
    N, M = A.shape[0], B.shape[0]
    iou3d = np.zeros((N, M), dtype=np.float32)
    iou_bev = np.zeros((N, M), dtype=np.float32)

    # for height overlap, since y face down, use the negative y
    min_h_a = -A[:, 0:4, 1].sum(axis=1) / 4.0
    max_h_a = -A[:, 4:8, 1].sum(axis=1) / 4.0
    min_h_b = -B[:, 0:4, 1].sum(axis=1) / 4.0
    max_h_b = -B[:, 4:8, 1].sum(axis=1) / 4.0

    for i in range(N):
        for j in range(M):
            max_of_min = np.max([min_h_a[i], min_h_b[j]])
            min_of_max = np.min([max_h_a[i], max_h_b[j]])	
            h_overlap = np.max([0, min_of_max - max_of_min])
            if h_overlap == 0:
                continue

            bottom_a, bottom_b = Polygon(A[i, 0:4, [0, 2]].T), Polygon(B[j, 0:4, [0, 2]].T)
            if bottom_a.is_valid and bottom_b.is_valid:
                # check is valid,  A valid Polygon may not possess any overlapping exterior or interior rings.
                bottom_overlap = bottom_a.intersection(bottom_b).area
            else:
                bottom_overlap = 0.
            overlap3d = bottom_overlap * h_overlap
            union3d = bottom_a.area * (max_h_a[i] - min_h_a[i]) + bottom_b.area * (max_h_b[j] - min_h_b[j]) - overlap3d
            iou3d[i][j] = overlap3d / union3d
            iou_bev[i][j] = bottom_overlap / (bottom_a.area + bottom_b.area - bottom_overlap)

    if need_bev:
        return iou3d, iou_bev

    return iou3d


def cls_type_to_id(cls_type):

    TYPE_ID_CONVERSION = {
        'Car': 0,
        'Cyclist': 1,
        'Pedestrian': 2,
    }
    if cls_type not in TYPE_ID_CONVERSION.keys():
        return -1
    return TYPE_ID_CONVERSION[cls_type]


class Calibration(object):
    def __init__(self, calib_file):
        if isinstance(calib_file, str):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def depthmap_to_rect(self, depth_map):
        """
        :param depth_map: (H, W), depth_map
        :return:
        """
        x_range = np.arange(0, depth_map.shape[1])
        y_range = np.arange(0, depth_map.shape[0])
        x_idxs, y_idxs = np.meshgrid(x_range, y_range)
        x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
        depth = depth_map[y_idxs, x_idxs]
        pts_rect = self.img_to_rect(x_idxs, y_idxs, depth)
        return pts_rect, x_idxs, y_idxs

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

    def camera_dis_to_rect(self, u, v, d):
        """
        Can only process valid u, v, d, which means u, v can not beyond the image shape, reprojection error 0.02
        :param u: (N)
        :param v: (N)
        :param d: (N), the distance between camera and 3d points, d^2 = x^2 + y^2 + z^2
        :return:
        """
        assert self.fu == self.fv, '%.8f != %.8f' % (self.fu, self.fv)
        fd = np.sqrt((u - self.cu)**2 + (v - self.cv)**2 + self.fu**2)
        x = ((u - self.cu) * d) / fd + self.tx
        y = ((v - self.cv) * d) / fd + self.ty
        z = np.sqrt(d**2 - x**2 - y**2)
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1)
        return pts_rect


class Object3d(object):

    TYPE_ID_CONVERSION = {
            'Car': 0,
            'Cyclist': 1,
            'Pedestrian': 2,
        }

    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = self.cls_type_to_id(self.cls_type)
        self.trucation = float(label[1])
        self.occlusion = int(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.ry = float(label[14])
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_obj_level()
    
    @property
    def dis_to_cam(self):
        return np.linalg.norm(self.loc)

    def get_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.trucation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 1  # Easy
        elif height >= 25 and self.trucation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 2  # Moderate
        elif height >= 25 and self.trucation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 3  # Hard
        else:
            self.level_str = 'UnKnown'
            return 4

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.loc
        return corners3d

    def to_str(self):
        print_str = '%s %.3f %d %.3f box2d: %s hwl: [%.3f %.3f %.3f] loc: %s ry: %.3f' \
                     % (self.cls_type, self.trucation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                        self.loc, self.ry)
        if self.score != -1.0:
            print_str += ' score: %.3f' % self.score
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.trucation, self.occlusion, self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2],
                       self.ry)
        if self.score != -1.0:
            kitti_str += ' %.2f' % self.score
        return kitti_str
    
    def to_pred_format(self):
        pred_str = '%s %f %d %f %f %f %f %f %f %f %f %f %f %f %f' \
                    % (self.cls_type, self.trucation, self.occlusion, self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2],
                       self.ry)
        if self.score != -1.0:
            pred_str += ' %f' % self.score
        else:
            pred_str += ' %f' % 1.0
        return pred_str

    def cls_type_to_id(self, cls_type):
        if cls_type not in self.TYPE_ID_CONVERSION.keys():
            return -1
        return self.TYPE_ID_CONVERSION[cls_type]