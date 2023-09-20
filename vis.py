import cv2
import numpy as np
def vis_points(frame, points, color):
    points = points.astype('int')
    for point in points:
        cv2.circle(frame, tuple(point), 2, color=color, thickness=-1)
    return frame


def vis_mask(frame, mask, color):
    overlay = np.zeros_like(frame)
    overlay[mask] = color
    frame = cv2.addWeighted(frame, 1, overlay, 0.75, 0)
    return frame
b=[5,4,3,2,1]
