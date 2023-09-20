import cv2
import numpy as np
from read_input import Input
from semantic_segmentation.semantic_segmentation import SemanticSegmentation
from object_detect import Mask_RCNN_detector

input_filename = '009.mp4'
input = Input(input_filename, preprocess=1)
video = input.read_video()

# ss = SemanticSegmentation('PSPNet', '../IPD/semantic_segmentation/pspnet101_cityscapes.caffemodel', 19)

lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([155, 50, 50])
upper_red2 = np.array([180, 255, 255])

skip = 0
count = 0


def vis_mask(frame, mask, color):
    overlay = np.zeros_like(frame)
    overlay[mask] = color
    frame = cv2.addWeighted(frame, 1, overlay, 0.75, 0)
    return frame


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

    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)
    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2
    radius = 150
    y, x = np.ogrid[:frame.shape[0], :frame.shape[1]]
    mask_circle = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    mask[mask_circle] = True
    result_image = cv2.add(image, mask1).astype('bool')
    return result_image


def to_road_mask(ss_mask):
    kernel = np.ones((5, 5), np.uint8)
    road_mask = np.where(ss_mask != 0, 0, 1).astype('uint8')
    dilated_road_mask = cv2.dilate(road_mask, kernel, iterations=3)
    return dilated_road_mask


def compare_ndarray(ndarray1, ndarray2, diff):
    return (ndarray1.astype('int16') - ndarray2.astype('int16')) > diff


while True:
    if count<600:
        ret, frame = video.read()
        count += 1
        continue
    ret, frame = video.read()
    frame = cv2.resize(frame, (720, 480), interpolation=cv2.INTER_AREA)
    frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 15)

    roi = getroi(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    average_brightness = cv2.mean(gray)[0]
    gray_threshold = (gray > 90)

    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1).astype('bool')
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2).astype('bool')

    red_bool_mask = np.logical_and(np.logical_or(red_mask1, red_mask2), np.logical_not(roi))
    red_bool_mask = cv2.morphologyEx(red_bool_mask.astype('uint8'), cv2.MORPH_CLOSE, np.ones((5,5), np.uint8)).astype(
        'bool')
    red_bool_mask = cv2.morphologyEx(red_bool_mask.astype('uint8'), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(
        'bool')
    rgb_check = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
    r_channel = frame[:, :, 2]
    g_channel = frame[:, :, 1]
    b_channel = frame[:, :, 0]


    # maskrgb = np.where(np.logical_and((r_channel > g_channel), (g_channel > b_channel)), 1, 0).astype('bool')
    # maskrbg = np.where(np.logical_and((r_channel > b_channel), (b_channel > g_channel)), 1, 0).astype('bool')
    # maskgbr = np.where(np.logical_and((g_channel > b_channel), (b_channel > r_channel)), 1, 0).astype('bool')
    # maskgrb = np.where(np.logical_and((g_channel > r_channel), (r_channel > b_channel)), 1, 0).astype('bool')
    # maskbgr = np.where(np.logical_and((b_channel > g_channel), (g_channel > r_channel)), 1, 0).astype('bool')
    # maskbrg = np.where(np.logical_and((b_channel > r_channel), (r_channel > g_channel)), 1, 0).astype('bool')
    # maskrg = np.where((r_channel > g_channel), 1, 0).astype('bool')
    # mask = (maskrgb.astype('int') + maskrbg.astype('int') + maskgbr.astype('int') +
    #         maskgrb.astype('int') + maskbgr.astype('int') + maskbrg.astype('int'))

    # frame = vis_mask(frame, maskrg, (0, 0, 240))
    # frame = vis_mask(frame, maskrgb, (0, 0, 255))
    # frame = vis_mask(frame, maskrbg, (255, 0, 255))
    # frame = vis_mask(frame, maskbgr, (255, 255, 0))
    # frame = vis_mask(frame, maskbrg, (255, 0, 0))

    # frame = vis_mask(frame, red_mask, (0, 0, 255))
    # ss_mask = np.array(ss.predict(frame))
    # road_bool_mask = to_road_mask(ss_mask).astype('bool')
    # redline = np.logical_and(red_bool_mask, road_bool_mask)
    # redline_roi = cv2.morphologyEx(red_bool_mask.astype('uint8'), cv2.MORPH_OPEN, np.ones((11, 11), np.uint8)).astype('bool')
    # redline_nroi = cv2.morphologyEx(red_bool_mask.astype('uint8'), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype('bool')
    # redline_nroi = cv2.morphologyEx(redline_nroi.astype('uint8'), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype('bool')
    # result_redline = np.where(roi, redline_roi, redline_nroi)
    # # frame = vis_mask(frame, roi, (0, 255, 0))
    # # frame = vis_mask(frame, road_bool_mask.astype('bool'), (255, 0, 0))
    # frame = vis_mask(frame, result_redline, (0, 0, 255))
    # printhsv = np.logical_and(red_bool_mask, roi)
    # for y in range(hsv.shape[0]):
    #     for x in range(hsv.shape[1]):
    #         if printhsv[y, x]:
    #             hsv_value = hsv[y, x]
    #             print(hsv_value)

    frame = vis_mask(frame, red_bool_mask, (255, 0, 255))

    cv2.imshow('r', frame)
    cv2.waitKey(1)
    print(count)
    count += 1
