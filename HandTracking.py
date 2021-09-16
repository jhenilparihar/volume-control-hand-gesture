import cv2
import mediapipe as mp
import math


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):

        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.hands = mp.solutions.hands.Hands(static_image_mode=self.mode, max_num_hands=self.max_hands,
                                              min_tracking_confidence=self.tracking_confidence,
                                              min_detection_confidence=self.detection_confidence)

        self.mpDraw = mp.solutions.drawing_utils
        self.landmark_list = []
        self.tipIds = [4, 8, 12, 16, 20]

    def find_hands(self, image, draw=True):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb).multi_hand_landmarks

        if self.results:
            for hand_landmark in self.results:
                if draw:
                    self.mpDraw.draw_landmarks(image, hand_landmark, mp.solutions.hands.HAND_CONNECTIONS)
        return image

    def find_position(self, image, hand_no=0, draw=True, draw_bounding_box=True):

        x_list = []
        y_list = []
        self.landmark_list = []
        bounding_box = []

        if self.results:
            hand = self.results[hand_no]
            for id_, landmark in enumerate(hand.landmark):
                h, w, c = image.shape
                cx, cy = int(landmark.x*w), int(landmark.y*h)
                x_list.append(cx)
                y_list.append(cy)
                self.landmark_list.append([id_, cx, cy])
                if draw:
                    cv2.circle(image, (cx, cy), 3, (20, 255, 0), cv2.FILLED)

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            bounding_box = x_min, y_min, x_max, y_max
            if draw_bounding_box:
                cv2.rectangle(image, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20),
                              (0, 255, 0), 2)

        return self.landmark_list, bounding_box

    def fingers_up(self):
        fingers = []
        # Thumb
        if self.landmark_list[self.tipIds[0]][1] > self.landmark_list[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id_ in range(1, 5):

            if self.landmark_list[self.tipIds[id_]][2] < self.landmark_list[self.tipIds[id_] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def find_distance(self, p1, p2, image, draw=True, r=10, t=3):
        x1, y1 = self.landmark_list[p1][1:]
        x2, y2 = self.landmark_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(image, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, image, [x1, y1, x2, y2, cx, cy]
