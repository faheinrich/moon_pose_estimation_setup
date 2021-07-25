import math
import time
import sys
import os
import os.path as osp
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
import torchvision

from posenet.main.config import cfg as posenet_cfg
from posenet.main.model import get_pose_net 
from posenet.data.dataset import generate_patch_image
from posenet.common.utils.pose_utils import process_bbox, pixel2cam

from rootnet.main.config import cfg as rootnet_cfg
from rootnet.main.model import get_pose_net as get_root_net
from rootnet.common.utils.pose_utils import process_bbox as rootnet_process_bbox
from rootnet.data.dataset import generate_patch_image as rootnet_generate_patch_image

from posenet.common.utils.vis import vis_keypoints, vis_3d_multiple_skeleton, vis_3d_multiple_skeleton_no_show_but_savefig

import rospy
from sensor_msgs.msg import Image, CameraInfo
import cv_bridge
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion
from std_msgs.msg import ColorRGBA, Header


"""
Paper:
Camera Distance-aware Top-down Approach for 3D Multi-person PoseEstimation from a Single RGB Image
"""


# MuCo joint set
joint_num = 21
joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )

# LOAD FASTER RCNN FOR BBOXES
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
detector_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, pretrained_backbone=True)
detector_model.eval().to(device)

detector_score_threshold = 0.8
detector_transform = transforms.Compose([
    transforms.ToTensor(),
])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args

args = parse_args()

posenet_cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

# LOAD POSENET
model_path_posenet = 'snapshot_24.pth.tar'
assert osp.exists(model_path_posenet), 'Cannot find model at ' + model_path_posenet
print('Load checkpoint from {}'.format(model_path_posenet))
posenet_model = get_pose_net(posenet_cfg, False, joint_num)
posenet_model = DataParallel(posenet_model).cuda()
ckpt = torch.load(model_path_posenet)
posenet_model.load_state_dict(ckpt['network'])
posenet_model.eval()
posenet_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=posenet_cfg.pixel_mean, std=posenet_cfg.pixel_std)])


#LOAD ROOTNET
model_path_rootnet = 'snapshot_18.pth.tar'
assert osp.exists(model_path_rootnet), 'Cannot find model at ' + model_path_rootnet
print('Load checkpoint from {}'.format(model_path_rootnet))
rootnet_model = get_root_net(rootnet_cfg, False)
rootnet_model = DataParallel(rootnet_model).cuda()
ckpt_rootnet = torch.load(model_path_rootnet)
rootnet_model.load_state_dict(ckpt_rootnet['network'])
rootnet_model.eval()
rootnet_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=rootnet_cfg.pixel_mean, std=rootnet_cfg.pixel_std)])



# SELECT CAMERA HIER

# camera specific, meine ms camera
original_img_height, original_img_width = (480, 640)
focal = [678, 678]
princpt = [318, 228]
camera_sub_topic = "/camera/image_raw"
CAMERA_INFO_TOPIC = '/camera/camera_info'


# labor aufnahme
# original_img_height, original_img_width = (960, 1280)
# focal = [524, 524]
# princpt = [320, 240]
# camera_sub_topic = "/camera/rgb/image_color"
# CAMERA_INFO_TOPIC = '/camera/rgb/camera_info'

def process_frame(original_img):

    whole_time = time.time()
    boxes_time = time.time()    

    # FASTER RCNN GET PERSON BBOXES
    model_input = detector_transform(original_img).unsqueeze(0).to(device)
    outputs = detector_model(model_input)
    labels = outputs[0]['labels'].cpu().detach().numpy()
    pred_scores = outputs[0]['scores'].cpu().detach().numpy()
    pred_bboxes = outputs[0]['boxes'].cpu().detach().numpy()
    bbox_list = pred_bboxes[pred_scores >= detector_score_threshold]
    labels = labels[pred_scores >= detector_score_threshold]
    bbox_list = bbox_list[labels==1]
    pose_time = time.time()

    # ROOTNET GET ESTIMATED ROOT DEPTH IN IMAGE
    person_num = len(bbox_list)
    root_depth_list = np.zeros(person_num)
    for n in range(person_num):
        bbox = rootnet_process_bbox(np.array(bbox_list[n]), original_img_width, original_img_height)
        img, img2bb_trans = rootnet_generate_patch_image(original_img, bbox, False, 0.0) 
        img = rootnet_transform(img).cuda()[None,:,:,:]
        k_value = np.array([math.sqrt(rootnet_cfg.bbox_real[0]*rootnet_cfg.bbox_real[1]*focal[0]*focal[1]/(bbox[2]*bbox[3]))]).astype(np.float32)
        k_value = torch.FloatTensor([k_value]).cuda()[None,:]
        # forward
        with torch.no_grad():
            root_3d = rootnet_model(img, k_value) # x,y: pixel, z: root-relative depth (mm)
        img = img[0].cpu().numpy()
        root_3d = root_3d[0].cpu().numpy()
        root_depth_list[n] = root_3d[2]


    if person_num < 1:
        return None

    # POSENET GET 3D POSE ESTIMATIO
    output_pose_2d = np.zeros((person_num, joint_num, 2))
    output_pose_3d = np.zeros((person_num, joint_num, 3))
    for n in range(person_num):
        bbox = process_bbox(np.array(bbox_list[n]), original_img_width, original_img_height)
        img, img2bb_trans = generate_patch_image(original_img, bbox, False, 1.0, 0.0, False) 
        img = posenet_transform(img).cuda()[None,:,:,:]

        # forward pass
        with torch.no_grad():
            pose_3d = posenet_model(img) # x,y: pixel, z: root-relative depth (mm)

        # inverse affine transform (restore the crop and resize)
        pose_3d = pose_3d[0].cpu().numpy()
        pose_3d[:,0] = pose_3d[:,0] / posenet_cfg.output_shape[1] * posenet_cfg.input_shape[1]
        pose_3d[:,1] = pose_3d[:,1] / posenet_cfg.output_shape[0] * posenet_cfg.input_shape[0]
        pose_3d_xy1 = np.concatenate((pose_3d[:,:2], np.ones_like(pose_3d[:,:1])),1)
        img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0,0,1]).reshape(1,3)))
        pose_3d[:,:2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1,0)).transpose(1,0)[:,:2]
        output_pose_2d[n] = pose_3d[:,:2]

        # root-relative discretized depth -> absolute continuous depth
        pose_3d[:,2] = (pose_3d[:,2] / posenet_cfg.depth_dim * 2 - 1) * (posenet_cfg.bbox_3d_shape[0]/2) + root_depth_list[n]
        pose_3d = pixel2cam(pose_3d, focal, princpt)
        output_pose_3d[n] = pose_3d


    print("time:%.4f," % (time.time() - whole_time), "boxes:%.4f," % (time.time() - boxes_time), "pose:%.4f" % (time.time() - pose_time))
    return output_pose_3d
    





    
color_frame = None

bridge = cv_bridge.CvBridge()

pub_pose = rospy.Publisher('/pose', MarkerArray)



class CameraCalibSubscriber():
    def __init__(self, camera_info_topic):
        self.subscriber = rospy.Subscriber(camera_info_topic,
                                        CameraInfo, self.camera_callback, queue_size=1)
        self.stop = False
        self.K = None
        self.camera_frame_id = None

    def camera_callback(self, data):
        self.K = np.reshape(np.array(data.K), [3, 3])
        self.camera_frame_id = data.header.frame_id
        self.stop = True

    def wait_for_calib(self):
        try:
            while not self.stop:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Shutting down")

        return self.K, self.camera_frame_id


print("hier")
# read calib from ros topic
camera_calib = CameraCalibSubscriber(CAMERA_INFO_TOPIC)
rospy.init_node('listener', anonymous=True)
K, camera_frame_id = camera_calib.wait_for_calib()
print(camera_frame_id)


def color_callback(msg):
    global color_frame, bridge
    color_frame = bridge.imgmsg_to_cv2(msg)
    


def ros_process():
    global color_frame

    if color_frame is None:
        rospy.rostime.wallsleep(0.03)
        return

    # color_frame = color_frame[::-1,:,...].copy()

    color = color_frame
    color_frame = None

    

    coord3d_mat = process_frame(color)
    if coord3d_mat is None:
        return

    coord3d_mat = coord3d_mat / 500

    ma = MarkerArray()
    h = Header(frame_id=camera_frame_id)
    line_list = Marker(type=Marker.LINE_LIST, id=0)
    line_list.header = h
    line_list.action = Marker.ADD
    line_list.scale.x = 0.05

    for pid in range(coord3d_mat.shape[0]):
        # # broadcast keypoints as tf
        # for kid in range(coord3d_mat.shape[1]):
        #     br.sendTransform((coord3d_mat[pid, kid, 0], coord3d_mat[pid, kid, 1], coord3d_mat[pid, kid, 2]),
        #          transformations.quaternion_from_euler(0, 0, 0),
        #          rospy.Time.now(),
        #          "/human_pose/person%d/%s" % (pid, KEYPOINT_NAME_DICT[kid]),
        #          camera_frame_id)

        # draw skeleton figure
        for lid, (p0, p1) in enumerate(skeleton):
            # if vis_mat[pid, p0] and vis_mat[pid, p1]:
            p0 = Point(x=coord3d_mat[pid, p0, 0],
                        y=coord3d_mat[pid, p0, 2],
                        z= -coord3d_mat[pid, p0, 1])
            p1 = Point(x=coord3d_mat[pid, p1, 0],
                        y=coord3d_mat[pid, p1, 2],
                        z= -coord3d_mat[pid, p1, 1])
            line_list.points.append(p0)
            line_list.points.append(p1)
    line_list.color.r = 1.0
    line_list.color.g = 0.0
    line_list.color.b = 0.0
    line_list.color.a = 1.0
    ma.markers.append(line_list)
    pub_pose.publish(ma)
    print("published pose")
    print(coord3d_mat.shape)




rospy.Subscriber(camera_sub_topic, Image, color_callback, queue_size=1)


try:
    while not rospy.core.is_shutdown():
        ros_process()
except KeyboardInterrupt:
    rospy.core.signal_shutdown('keyboard interrupt')

# cv2.destroyAllWindows()