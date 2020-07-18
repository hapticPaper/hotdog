import os, time, numpy as np
import cv2

os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF']='0'

class fastCapture:

    def __init__(self):
        pass

    def captureS(self, frames=24, path='./', prefix='image_', file_type='png', sleep_duration=2):
        vc = cv2.VideoCapture(0)
        for i in range(0,frames):
            ret, frame = vc.read()
            cv2.imwrite(os.path.join(path, f"{prefix}{int(time.time())}.{file_type}"), frame)
            time.sleep(sleep_duration)
        vc.release()
        cv2.imshow('frame', frame)


if __name__=="__main__":
    c = fastCapture()
    c.captureS(5, path='nodog',sleep_duration=0)