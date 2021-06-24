import cv2
import mediapipe as mp

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



# import tensorflow_hub as hub
# import tensorflow as tf

def main():
    """
    Camera Distance-aware Top-down Approach for 3D Multi-person PoseEstimation from a Single RGB Image
    """

    

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
        # parser.add_argument('--test_epoch', type=str, dest='test_epoch')
        args = parser.parse_args()

        # test gpus
        if not args.gpu_ids:
            assert 0, print("Please set proper gpu ids")

        if '-' in args.gpu_ids:
            gpus = args.gpu_ids.split('-')
            gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
            gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
            args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
        
        # assert args.test_epoch, 'Test epoch is required.'
        return args

    # argument parsing
    args = parse_args()

    posenet_cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True


    # MuCo joint set
    joint_num = 21
    joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
    flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
    skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )

    # snapshot load posenet
    model_path_posenet = posenet_path + '/demo/snapshot_%d.pth.tar' % int(24)
    assert osp.exists(model_path_posenet), 'Cannot find model at ' + model_path_posenet
    print('Load checkpoint from {}'.format(model_path_posenet))
    model = get_pose_net(posenet_cfg, False, joint_num)
    model = DataParallel(model).cuda()
    ckpt = torch.load(model_path_posenet)
    model.load_state_dict(ckpt['network'])
    model.eval()


    # snapshot load rootnet
    model_path_rootnet = rootnet_path + '/demo/snapshot_%d.pth.tar' % int(18)
    assert osp.exists(model_path_rootnet), 'Cannot find model at ' + model_path_rootnet
    print('Load checkpoint from {}'.format(model_path_rootnet))
    rootnet_model = get_root_net(rootnet_cfg, False)
    rootnet_model = DataParallel(rootnet_model).cuda()
    ckpt_rootnet = torch.load(model_path_rootnet)
    rootnet_model.load_state_dict(ckpt_rootnet['network'])
    rootnet_model.eval()


    rootnet_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=rootnet_cfg.pixel_mean, std=rootnet_cfg.pixel_std)])
    posenet_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=posenet_cfg.pixel_mean, std=posenet_cfg.pixel_std)])

    original_img_height, original_img_width = (480, 640)



    # normalized camera intrinsics
    # focal = [1500, 1500] # x-axis, y-axis
    focal = [678, 678]
    princpt = [318, 228]
    # princpt = [original_img_width/2, original_img_height/2] # x-axis, y-axis
    # print('focal length: (' + str(focal[0]) + ', ' + str(focal[1]) + ')')
    # print('principal points: (' + str(princpt[0]) + ', ' + str(princpt[1]) + ')')


    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        
        # print("+++++++++++++++++")
        
        success, original_img = cap.read()
        if not success:
            print("Webcam failed somehow?")
            continue


        whole_time = time.time()

        # get bboxes
        boxes_time = time.time()
        

        model_input = detector_transform(original_img).unsqueeze(0).to(device)
        outputs = detector_model(model_input)
    
        pred_scores = outputs[0]['scores'].cpu().detach().numpy()
        pred_bboxes = outputs[0]['boxes'].cpu().detach().numpy()
        bbox_list = pred_bboxes[pred_scores >= detector_score_threshold]

        pose_time = time.time()




        # calculate roots
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



        # assert len(bbox_list) == len(root_depth_list)
  

        if person_num < 1:
            continue



        # for each cropped and resized human image, forward it to PoseNet
        output_pose_2d = np.zeros((person_num, joint_num, 2))
        output_pose_3d = np.zeros((person_num, joint_num, 3))
        for n in range(person_num):
            bbox = process_bbox(np.array(bbox_list[n]), original_img_width, original_img_height)
            img, img2bb_trans = generate_patch_image(original_img, bbox, False, 1.0, 0.0, False) 
            img = posenet_transform(img).cuda()[None,:,:,:]

            # forward
            with torch.no_grad():
                pose_3d = model(img) # x,y: pixel, z: root-relative depth (mm)

            # inverse affine transform (restore the crop and resize)
            pose_3d = pose_3d[0].cpu().numpy()
            pose_3d[:,0] = pose_3d[:,0] / posenet_cfg.output_shape[1] * posenet_cfg.input_shape[1]
            pose_3d[:,1] = pose_3d[:,1] / posenet_cfg.output_shape[0] * posenet_cfg.input_shape[0]
            pose_3d_xy1 = np.concatenate((pose_3d[:,:2], np.ones_like(pose_3d[:,:1])),1)
            img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0,0,1]).reshape(1,3)))
            pose_3d[:,:2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            
            # root-relative discretized depth -> absolute continuous depth
            pose_3d[:,2] = (pose_3d[:,2] / posenet_cfg.depth_dim * 2 - 1) * (posenet_cfg.bbox_3d_shape[0]/2) + root_depth_list[n]
            pose_3d = pixel2cam(pose_3d, focal, princpt)
            output_pose_3d[n] = pose_3d


        print("time:%.4f," % (time.time() - whole_time), "boxes:%.4f," % (time.time() - boxes_time), "pose:%.4f" % (time.time() - pose_time))

        cv2.imwrite("frame.jpg", original_img)
        np.save("result3d.npy", output_pose_3d)
        

    cap.release()


if __name__ == "__main__":
    main()


