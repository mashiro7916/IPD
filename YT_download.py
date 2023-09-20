from pytube import YouTube
import os
import cv2

class YTdownloader:
    def __init__(self):
        last_existing_filename = max(os.listdir('/media/jacky72503/data/IPD/data/frame'))
        next_number = int(last_existing_filename) + 1
        self.new_filename = "{:03d}".format(next_number)
        self.output_video_path = os.path.join('/media/jacky72503/data/IPD/data/video/')
        self.output_frame_path = os.path.join('/media/jacky72503/data/IPD/data/frame/', self.new_filename)


    def download(self, url, toframes=True):
        yt = YouTube(url)
        yt.streams.filter().get_highest_resolution().download(output_path=self.output_video_path,
                                                              filename=self.new_filename+'.mp4')
        if toframes:
            self.video2frames(os.path.join(self.output_video_path,self.new_filename+'.mp4'))
    def video2frames(self,video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        if not os.path.exists(self.output_frame_path):
            os.makedirs(self.output_frame_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = f"frame{frame_count:04d}.png"
            cv2.imwrite(os.path.join(self.output_frame_path+'/',frame_filename), frame)
            frame_count += 1
        cap.release()



ytd = YTdownloader()
ytd.download('https://www.youtube.com/watch?v=EAUsJbNKOZg')
