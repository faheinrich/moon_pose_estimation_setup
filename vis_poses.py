import cv2
import mediapipe as mp

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import math
import time

import sys
import os
import os.path as osp

import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

posenet_path = os.getcwd() + "/posenet"
sys.path.insert(0, osp.join(posenet_path, 'main'))
sys.path.insert(0, osp.join(posenet_path, 'data'))
sys.path.insert(0, osp.join(posenet_path, 'common'))

from posenet.main.config import cfg as posenet_cfg
from posenet.main.model import get_pose_net 
from posenet.data.dataset import generate_patch_image
from posenet.common.utils.pose_utils import process_bbox, pixel2cam

import matplotlib.pyplot as plt


rootnet_path = os.getcwd() + "/rootnet"
sys.path.insert(0, osp.join(rootnet_path, 'main'))
sys.path.insert(0, osp.join(rootnet_path, 'data'))
sys.path.insert(0, osp.join(rootnet_path, 'common'))
from rootnet.main.config import cfg as rootnet_cfg
from rootnet.main.model import get_pose_net as get_root_net
from rootnet.common.utils.pose_utils import process_bbox as rootnet_process_bbox
from rootnet.data.dataset import generate_patch_image as rootnet_generate_patch_image

import torchvision

from posenet.common.utils.vis import vis_keypoints, vis_3d_multiple_skeleton, vis_3d_multiple_skeleton_no_show_but_savefig

skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )
joint_num = 21
import time

while True:
    time.sleep(0.05)
    try:
        
        output_pose_3d = np.load("result3d.npy")
        print(output_pose_3d.shape)

        output_pose_2d = output_pose_3d[:,:,:2]

        person_num = output_pose_3d.shape[0]

        # visualize 2d poses
        vis_img = cv2.imread("frame.jpg")
        for n in range(person_num):
            vis_kps = np.zeros((3,joint_num))
            vis_kps[0,:] = output_pose_2d[n][:,0]
            vis_kps[1,:] = output_pose_2d[n][:,1]
            vis_kps[2,:] = 1
            vis_img = vis_keypoints(vis_img, vis_kps, skeleton)
        cv2.imshow('vis img', cv2.resize(vis_img, (vis_img.shape[1]*2,vis_img.shape[0]*2 )))

        # # visualize 3d poses
        # vis_kps = np.array(output_pose_3d)
        # vis_3d_multiple_skeleton_no_show_but_savefig(vis_kps, np.ones_like(vis_kps), skeleton, 'output_pose_3d (x,y,z: camera-centered. mm.)')


        # vis_img = cv2.imread("poses.png")[:,:,::-1]


        # cv2.imshow("poses und so", cv2.resize(vis_img, (vis_img.shape[1]*2,vis_img.shape[0]*2)))
        if cv2.waitKey(1) & 0xFF == 27:
            break
    except Exception:
        print("wtf")