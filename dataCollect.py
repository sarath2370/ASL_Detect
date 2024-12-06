import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

imgSize = 300
offset = 20
counter = 0

folder = "SignData/D"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        imgCropShape = imgCrop.shape

        aspect_ratio = h / w

        if aspect_ratio > 1:
            k = imgSize / h
            wCalc = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCalc, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCalc) / 2)
            imgWhite[0:imgResizeShape[0], wGap:wCalc+wGap] = imgResize
        else:
            k = imgSize / h
            hCalc = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (hCalc, imgSize))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCalc) / 2)
            imgWhite[hGap:hCalc + hGap, 0:imgResizeShape[1]] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("imageWhite", imgWhite)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
