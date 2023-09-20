import sys

sys.path.append('../IPD/MOD')
from lightglue.superpoint import SuperPoint
from lightglue.lightglue import LightGlue
from lightglue.disk import DISK
from lightglue.utils import numpy_image_to_torch, rbd
import cv2
import numpy as np
import torch
import random


class CorrespondPoints:
    def __init__(self, extractor, max_num_keypoints, device='cuda'):
        self.device = device
        if extractor == 'superpoint':
            self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
            self.matcher = LightGlue(features='superpoint').eval().to(device)
        elif extractor == 'disk':
            self.extractor = DISK(max_num_keypoints=max_num_keypoints).eval().cuda()
            self.matcher = LightGlue(features='disk').eval().cuda()

    def getmatch(self, img1, img2):
        with torch.no_grad():
            img1 = numpy_image_to_torch(img1[..., ::-1])
            img2 = numpy_image_to_torch(img2[..., ::-1])
            feats0 = self.extractor.extract(img1.to(self.device))
            feats1 = self.extractor.extract(img2.to(self.device))
            matches = self.matcher({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches = [rbd(x) for x in [feats0, feats1, matches]]
        return feats0, feats1, matches

    def vis(self, kpts0, kpts1, matches, frame):
        for i in range(len(kpts1)):
            coordinate = tuple(np.round(np.array(kpts1[i].cpu())).astype(int))
            if i in np.array(matches[..., 1].cpu()):
                cv2.circle(frame, coordinate, 1, color=(255, 0, 0), thickness=-1)
            else:
                cv2.circle(frame, coordinate, 1, color=(0, 0, 255), thickness=-1)
        return frame

    def find_outlier(self, pre_points, cur_points):
        M, mask = cv2.findHomography(pre_points, cur_points, cv2.RANSAC, 2, confidence=0.999)
        inlier = cur_points[np.squeeze(mask) == 1]
        outlier = cur_points[np.squeeze(mask) == 0]
        return M, inlier, outlier

    def vis_inoutlier(self, frame, inlier, outlier):
        for i in range(len(inlier)):
            coordinate = tuple(np.round(np.array(inlier[i])).astype(int))
            cv2.circle(frame, coordinate, 3, color=(255, 0, 0), thickness=-1)
        for i in range(len(outlier)):
            coordinate = tuple(np.round(np.array(outlier[i])).astype(int))
            cv2.circle(frame, coordinate, 3, color=(0, 0, 255), thickness=-1)
        return frame


# CP = CorrespondPoints('superpoint', 2048)
# for i in range(500):
#     img1 = cv2.imread(f'/media/jacky72503/data/IPD/data/frame/001/filename{i + 1:03d}.png')
#     img2 = cv2.imread(f'/media/jacky72503/data/IPD/data/frame/001/filename{i + 4:03d}.png')
#     feats0, feats1, matches = CP.getmatch(img1, img2)
#     kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches['matches']
#     m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
#     p = m_kpts0.cpu().numpy()
#     m_kpts0, m_kpts1 = np.array(m_kpts0.cpu()), np.array(m_kpts1.cpu())
#     M, inlier, outlier = CP.find_outlier(m_kpts0, m_kpts1)
#     result = CP.vis_inoutlier(img2, inlier, outlier)
#     cv2.imshow('f',result)
#     cv2.waitKey(1)
# img1 = cv2.imread(f'/media/jacky72503/data/IPD/data/frame/001/filename010.png')
# img2 = cv2.imread(f'/media/jacky72503/data/IPD/data/frame/001/filename013.png')
# img3 = cv2.imread(f'/media/jacky72503/data/IPD/data/frame/001/filename016.png')
# feats01, feats02, matches0 = CP.getmatch(img1, img2)
# feats11, feats12, matches1 = CP.getmatch(img2, img3)
# kpts01, kpts02, matches0 = feats01['keypoints'], feats02['keypoints'], matches0['matches']
# kpts11, kpts12, matches1 = feats11['keypoints'], feats12['keypoints'], matches1['matches']
# print(kpts02.shape,kpts11.shape)
# print(kpts02)
# print(kpts11)
# print(kpts11==kpts02)
# a = 0
# b = 0

# cv2.imwrite(f'/home/jacky72503/PycharmProjects/IPD/result/disk/r{i}.png', r)


# feats0, feats1, matches = CP.getmatch(img1, img2)
# kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches['matches']
# m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
# r = CP.vis(kpts0, kpts1, matches, img2)
# print(len(kpts1),len(matches))
# cv2.imwrite('r.png', r)
# # print(kpts0)
# # print(kpts1)
# print(matches)
