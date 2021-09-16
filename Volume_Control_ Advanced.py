import cv2
import time
import numpy as np
import HandTracking as ht
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


#############################################################

cam_width, cam_height = 640, 480
previous_time = 0
volume_per = 0
volume_bar = 0
vol = 400
color_vol = (255, 0, 0)
count = 0

#############################################################

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

detector = ht.HandDetector(detection_confidence=0.7, max_hands=1)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volume_range = volume.GetVolumeRange()
min_vol = volume_range[0]
max_vol = volume_range[1]

while True:
    success, img = cap.read()

    # ---------------------------------------- #
    #              Find Hand                   #
    # ---------------------------------------- #

    img = detector.find_hands(img)
    landmark_list, bounding_box = detector.find_position(img, draw=False)

    if len(landmark_list) != 0:

        # ---------------------------------------- #
        #           Filter based on size           #
        # ---------------------------------------- #

        area = (bounding_box[2]-bounding_box[0]) * (bounding_box[3]-bounding_box[1])//100
        # print(area)

        if 100 < area < 1500:
            # print(f"yes {area}")

            # ---------------------------------------- #
            #   Find Distance between Index and Thumb  #
            # ---------------------------------------- #

            length, img, line_info = detector.find_distance(4, 8, img)

            # ---------------------------------------- #
            #               Convert Volume             #
            # ---------------------------------------- #

            # Hand Range 50 - 300
            # Volume Range -74 - 0

            # print(f"{vol} : {length}")
            # volume.SetMasterVolumeLevel(vol, None)
            volume_bar = np.interp(length, [20, 180], [400, 150])
            volume_per = np.interp(length, [20, 180], [0, 100])

            # ---------------------------------------- #
            #   Reduce Resolution to make it smoother  #
            # ---------------------------------------- #

            smoothness = 10
            volume_per = smoothness * round(volume_per/smoothness)

            # ---------------------------------------- #
            #            Check finger up               #
            # ---------------------------------------- #

            fingers = detector.fingers_up()
            # print(fingers)

            # ---------------------------------------- #
            #       If pinky is down set volume        #
            # ---------------------------------------- #

            if not fingers[3]:
                volume.SetMasterVolumeLevelScalar(volume_per / 100, None)
                cv2.circle(img, (line_info[4], line_info[5]), 10, (20, 255, 0), cv2.FILLED)
                color_vol = (0, 255, 0)
            else:
                color_vol = (255, 0, 0)
            # print(landmark_list[4], landmark_list[8])
            # print(length)
            count += 1
        # else:
        #     print(f"no {area}")

            # ---------------------------------------- #
            #               Drawings                   #
            # ---------------------------------------- #

    if count > 0:
        cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 0), 2)
        cv2.rectangle(img, (51, int(volume_bar)), (84, 400), (255, 0, 0), cv2.FILLED)

        cv2.putText(img, text=f"{int(volume_per)}%", org=(40, 450), fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=2, color=(255, 0, 0), thickness=3)

    current_volume = int(volume.GetMasterVolumeLevelScalar()*100)
    cv2.putText(img, text=f"Volume Set : {current_volume}", org=(350, 50), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2,
                color=color_vol, thickness=3)

    # ---------------------------------------- #
    #               Frame rate                 #
    # ---------------------------------------- #

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, text=f"FPS : {int(fps)}", org=(10, 50), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2,
                color=(255, 0, 0), thickness=3)

    cv2.imshow("Img", img)
    cv2.waitKey(1)
