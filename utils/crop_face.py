import os
import cv2
from moviepy.editor import *
import face_recognition

def crop_face(path):
    for video in range(32): # one subject has 32 video trials
        print("process video: ", video)
        video_path = os.path.join(path, str(video)+".mp4")
        save_path = os.path.join(path, str(video))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cap = cv2.VideoCapture(video_path)
        frames_num = cap.get(7)
        rate = cap.get(5)
        duration = round(frames_num / rate)
        # print(frames_num, rate, duration)

        # main loop reading each frame of video
        count = 0
        ret = True
        pre_face_locations = [(0, 640, 480, 0 )]
        while ret:
            ret, frame = cap.read()
            if (count % int(rate)) == 0 and count < frames_num:
                face_locations = face_recognition.face_locations(frame)
                if len(face_locations) > 0:
                    crop_frame = frame[face_locations[0][0]:face_locations[0][2], face_locations[0][3]:face_locations[0][1],:]
                    pre_face_locations = face_locations
                else:
                    print("video: ", video, " frame: ", count, "detect fail, use pre_face_location ")
                    crop_frame = frame[pre_face_locations[0][0]:pre_face_locations[0][2], pre_face_locations[0][3]:pre_face_locations[0][1],:]
                # print(crop_frame)
                cv2.imwrite(os.path.join(save_path, "{}.jpg".format(int(count / rate))), crop_frame)
               
            count = count + 1


# for all sujects
if __name__ == '__main__':
    path = './MixedEmoR'
    for subject in range(35):
        print("process subject ", str(subject + 1))
        crop_face(os.path.join(path, str(subject+1)))

