import cv2
import sys
import numpy as np
import os
from MOD.track import TrackObject
from vis import vis_points, vis_mask
from MOD.correspond_points import CorrespondPoints
from read_input import Input
from semantic_segmentation.semantic_segmentation import SemanticSegmentation
from object_detect import Mask_RCNN_detector
from Eliminate_duplicates import eliminate_duplicates
import time
import torch
import os

input_filename = '001.mp4'
output_path = os.path.join('/media/jacky72503/data/IPD/detect_result/', input_filename.split('.')[0])

if not os.path.exists(output_path):
    os.makedirs(output_path)

result_output_path = os.path.join('/media/jacky72503/data/IPD/full_result/', input_filename.split('.')[0])
if not os.path.exists(result_output_path):
    os.makedirs(result_output_path)

input = Input(input_filename, preprocess=2)
video = input.read_video()

print('Loading semantic segmentation pretrained weights')
ss = SemanticSegmentation('PSPNet', '../IPD/semantic_segmentation/pspnet101_cityscapes.caffemodel', 19)
print('Successfully load ')
cp = CorrespondPoints('superpoint', 1024)
od = Mask_RCNN_detector()

lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([155, 50, 50])
upper_red2 = np.array([180, 255, 255])


def getroi(frame):
    image = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    point1 = ((image.shape[1] / 2) - 40, 0)
    point2 = (0, image.shape[0])
    point3 = (int(image.shape[1] / 2), image.shape[0])

    mask0 = np.zeros(image.shape, dtype=np.uint8)
    triangle = np.array([point1, point2, point3], dtype=np.int32)
    cv2.fillPoly(mask0, [triangle], 255)

    vertices = [(340, 380), (340, 425), (700, 425), (700, 380)]
    mask1 = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    rect = np.array(vertices, dtype=np.int32)
    cv2.fillPoly(mask1, [rect], 255)

    result_image = cv2.add(image, mask1).astype('bool')
    return result_image


def get_points_in_mask(points, mask_coordinate):
    points = np.round(points).astype('int')
    matching_rows = []
    for points_pair in points:
        matching_row = np.where((mask_coordinate[:, 1] == points_pair[0]) & (mask_coordinate[:, 2] == points_pair[1]))[
            0]
        if len(matching_row) > 0:
            matching_rows.append(mask_coordinate[matching_row[0]])
    matching_rows = np.array(matching_rows)

    return matching_rows


def to_road_mask(ss_mask):
    kernel = np.ones((5, 5), np.uint8)
    road_mask = np.where(ss_mask != 0, 0, 1).astype('uint8')
    dilated_road_mask = cv2.dilate(road_mask, kernel, iterations=3)
    return dilated_road_mask


def count_value_in_ndarray(ndarray):
    u, c = np.unique(ndarray, return_counts=True)
    print(u, c)


def detect_illegal(redline, road, all_objects, cars, check_moving, frame):
    redline = np.logical_and(redline,
                             cv2.dilate(road.astype('uint8'), np.ones((11, 11), np.uint8), iterations=1).astype('bool'))
    object_mask = np.logical_or.reduce(all_objects, axis=0)
    redline = np.logical_and(np.logical_and(np.logical_not(object_mask), redline), np.logical_not(roi))
    # redline = np.logical_and(np.logical_not(object_mask), redline)
    redline = cv2.morphologyEx(redline.astype('uint8'), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype('bool')
    redline = cv2.morphologyEx(redline.astype('uint8'), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)).astype('bool')

    cars = cars[np.logical_not(check_moving)]
    objects_in_road = []

    linesP = cv2.HoughLinesP(redline.astype('uint8') * 255, 1, np.pi / 180, 50, None, 70, 10)
    extension_length = 30
    redline_extension = np.zeros_like(redline, dtype=np.uint8)
    if linesP is not None:
        for line in linesP:
            x1, y1, x2, y2 = line[0]
            direction_vector = np.array([x2 - x1, y2 - y1])
            normalized_direction = direction_vector / np.linalg.norm(direction_vector)

            new_x1 = int(x1 - extension_length * normalized_direction[0])
            new_y1 = int(y1 - extension_length * normalized_direction[1])
            new_x2 = int(x2 + extension_length * normalized_direction[0])
            new_y2 = int(y2 + extension_length * normalized_direction[1])

            cv2.line(redline_extension, (new_x1, new_y1), (new_x2, new_y2), (255, 255, 255), 1)
    redline_extension = redline_extension.astype('bool')
    for car in cars:
        if np.any(np.logical_and(car, road)):
            objects_in_road.append(car)

    frame = vis_mask(frame, road, (255, 0, 0))
    frame = vis_mask(frame, redline_extension, (0, 0, 255))
    result = []
    for o in objects_in_road:
        if np.any(np.logical_and(redline_extension, o)):
            result.append(o)
            frame = vis_mask(frame, o, (255, 0, 255))
            continue
        frame = vis_mask(frame, o, (0, 255, 0))
    # cv2.imshow('r', frame)
    # cv2.waitKey(1)
    return frame, result


skip = 0
start = time.time()
ss_total_time = 0
od_total_time = 0
check_moving_total_time = 0
check_illegal_total_time = 0
count = 0
ret, preframe = video.read()
preframe = cv2.resize(preframe, (720, 480), interpolation=cv2.INTER_AREA)
preframe = cv2.fastNlMeansDenoisingColored(preframe, None, 10, 10, 7, 15)
roi = getroi(preframe)
while True:

    ret, curframe = video.read()
    if not ret:
        break
    curframe = cv2.resize(curframe, (720, 480), interpolation=cv2.INTER_AREA)
    curframe = cv2.fastNlMeansDenoisingColored(curframe, None, 10, 10, 7, 15)
    frame1 = preframe
    frame2 = curframe

    od_start = time.time()
    mask_result = od.predict(frame2)
    cars_mask = od.post_process(mask_result, [3]).cpu().numpy()
    cars_temp = []
    for c in cars_mask:
        if np.any(c):
            cars_temp.append(c)
    cars_mask = np.array(cars_temp)
    all_objects = od.post_process(mask_result, [1, 2, 3, 4]).cpu().numpy()
    del mask_result
    od_end = time.time()
    od_total_time += od_end - od_start

    mask_coordinate = np.argwhere(cars_mask)[:, [0, 2, 1]]
    if len(mask_coordinate) > 0:
        object_count = np.max(mask_coordinate, axis=0)[0]
    check_moving_start = time.time()
    feats0, feats1, matches = cp.getmatch(frame1, frame2)
    kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches['matches']
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    m_kpts0, m_kpts1 = m_kpts0.cpu().numpy(), m_kpts1.cpu().numpy()
    M, inlier, outlier = cp.find_outlier(m_kpts0, m_kpts1)
    inlier_in_object_mask = get_points_in_mask(inlier, mask_coordinate)
    outlier_in_object_mask = get_points_in_mask(outlier, mask_coordinate)
    moving = []

    for i in range(object_count + 1):
        if len(inlier_in_object_mask) == 0 or len(outlier_in_object_mask) == 0:
            break
        in_count = len(inlier_in_object_mask[inlier_in_object_mask[:, 0] == i])
        out_count = len(outlier_in_object_mask[outlier_in_object_mask[:, 0] == i])
        if in_count + out_count <= 3:
            moving.append(1)
            continue
        moving.append(out_count / (in_count + out_count))
    moving = np.array(moving) > 0.1
    check_moving_end = time.time()
    check_moving_total_time += check_moving_end - check_moving_start

    # ------------------------------------------------------------
    hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = red_mask2 + red_mask1
    red_bool_mask = red_mask == 255

    ss_start = time.time()
    ss_mask = np.array(ss.predict(frame2))
    ss_end = time.time()
    ss_total_time += ss_end - ss_start
    road_bool_mask = to_road_mask(ss_mask).astype('bool')

    check_illegal_start = time.time()
    result_frame, illegal = detect_illegal(red_bool_mask, road_bool_mask, all_objects, cars_mask, moving, frame2)
    check_illegal_end = time.time()
    check_illegal_total_time += check_illegal_end - check_illegal_start
    result_frame = vis_points(result_frame,inlier,(255,255,0))
    result_frame = vis_points(result_frame, outlier, (0, 0, 255))
    cv2.imwrite(os.path.join(result_output_path, 'result_frame{}.png'.format(count)), result_frame)
    for i in range(len(illegal)):
        rmin, rmax, cmin, cmax = od.get_bbox(illegal[i], 20)
        if rmin < 0:
            rmin = 0
        if rmax > 480:
            rmax = 480
        if cmin < 0:
            cmin = 0
        if cmax > 720:
            cmax = 720
        print(rmin, rmax, cmin, cmax)
        output = frame2[rmin:rmax, cmin:cmax]
        cv2.imwrite(os.path.join(output_path, 'illega_object_for_frame{}_object{}.png'.format(count, i)), output)
    print(count, torch.cuda.mem_get_info())
    count += 1
    preframe = curframe
    # if count > len(frames) - 2:
    #     break
end = time.time()
count += 1
print('Process {} frames'.format(count))
print('for {} seconds for od,{} per second'.format(od_total_time, od_total_time / count))
print('for {} seconds for check moving,{} per second'.format(check_moving_total_time, check_moving_total_time / count))
print('for {} seconds for ss,{} per second'.format(ss_total_time, ss_total_time / count))
print(
    'for {} seconds for check illegal,{} per second'.format(check_illegal_total_time, check_illegal_total_time / count))
print('{} frames per second'.format(count / (end - start)))

final_result = eliminate_duplicates(input_filename.split('.')[0])
print(len(final_result))
output_path = os.path.join('/media/jacky72503/data/IPD/final_result/', input_filename.split('.')[0])
if not os.path.exists(output_path):
    os.makedirs(output_path)
for i, fr in enumerate(final_result):
    cv2.imwrite(os.path.join(output_path, 'illega_object_object{}.png'.format(i)), fr)
