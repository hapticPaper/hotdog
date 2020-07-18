import cv2, os, time

class camControl:
    def __init__(self):
        pass

    def captureN(self, n_frames, path='./', prefix='image_', file_type='png', sleep_duration=2):
        os.makedirs(path, exist_ok=True)
        cam = cv2.VideoCapture(0)
        for i in range(0,n_frames):
            return_value, image = cam.read()
            if i>0:
                cv2.imwrite(os.path.join(path, f"{prefix}{int(time.time())}.{file_type}"), image)
            time.sleep(sleep_duration)
        del(cam)


if __name__=="__main__":
    cam = camControl()
    cam.captureN(3,path='dog',sleep_duration=0)