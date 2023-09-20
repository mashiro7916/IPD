import cv2
import os


class Input:
    def __init__(self, file_name, preprocess=0):
        self.video_path = '/media/jacky72503/data/IPD/data/video'
        self.frame_path = '/media/jacky72503/data/IPD/data/frame'
        self.file_name = file_name
        self.preprocess = preprocess

    def read_video(self):
        path = os.path.join(self.video_path, self.file_name)
        print('read video from:', path)
        return cv2.VideoCapture(path)

    def read_frame(self):
        path = os.path.join(self.frame_path, self.file_name.split('.')[0])
        print('read frame from path:', path)
        filenames = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        frames = []
        print(len(filenames))
        for i in range(len(filenames)):
            print(i)
            if self.preprocess == 2:
                frame = self.preprocess2(cv2.imread(os.path.join(path, filenames[i])))
            elif self.preprocess == 1:
                frame = self.preprocess1(cv2.imread(os.path.join(path, filenames[i])))
            else:
                frame = cv2.imread(os.path.join(path, filenames[i]))
            frames.append(frame)
        return frames


    def preprocess1(self, frame):
        frame = cv2.resize(frame, (720, 480), interpolation=cv2.INTER_AREA)
        return frame
    def preprocess2(self, frame):
        frame = self.preprocess1(frame)
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 15)
        return frame
