import math

import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QImage, qRgb
import HandTrackingModule as ht
import autopy
import time
####################灰度图和二值化########################################

def gray_picture(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_gray

def qimage2mat(qtpixmap):    #qtpixmap转opencv
    qimg = qtpixmap.toImage()
    temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
    temp_shape += (4,)
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    result = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
    result = result[..., :3]
    return result


def matqimage(cvimg):       #opencv转QImage
    if cvimg.ndim==2:              #单通道
        height, width= cvimg.shape
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
        cvimg = QImage(cvimg.data, width, height, QImage.Format_RGB888)
        pix = QPixmap.fromImage(cvimg)
        return pix
    else:                          #多个通道
        width = cvimg.shape[1]
        height = cvimg.shape[0]
        pixmap = QPixmap(width, height)  # 根据已知的高度和宽度新建一个空的QPixmap,
        qimg = pixmap.toImage()         # 将pximap转换为QImage类型的qimg
        for row in range(0, height):
            for col in range(0, width):
                b = cvimg[row, col, 0]
                g = cvimg[row, col, 1]
                r = cvimg[row, col, 2]
                pix = qRgb(r, g, b)
                qimg.setPixel(col, row, pix)
                pix = QPixmap.fromImage(qimg)
        return pix

###############图像翻转####################################################
def flip_picture(image,x):    #图像  水平翻转:0   垂直翻转:1  沿xy轴翻转:-1
    image = cv2.flip(image,x)
    return image

def hand_tracking_mouse_control():
    pTime = 0
    width, height = 640, 480
    frameR = 100
    smoothening = 8
    prev_x, prev_y = 0, 0
    curr_x, curr_y = 0, 0

    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    detector = ht.handDetector(maxHands=1)
    screen_width, screen_height = autopy.screen.size()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist, bbox = detector.findPosition(img)

        if len(lmlist) != 0:
            x1, y1 = lmlist[8][1:]
            x2, y2 = lmlist[12][1:]

            fingers = detector.fingersUp()
            cv2.rectangle(img, (frameR, frameR), (width - frameR, height - frameR), (255, 0, 255), 2)

            if fingers[1] == 1 and fingers[2] == 0:
                x3 = np.interp(x1, (frameR, width - frameR), (0, screen_width))
                y3 = np.interp(y1, (frameR, height - frameR), (0, screen_height))

                curr_x = prev_x + (x3 - prev_x) / smoothening
                curr_y = prev_y + (y3 - prev_y) / smoothening

                autopy.mouse.move(screen_width - curr_x, curr_y)
                cv2.circle(img, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
                prev_x, prev_y = curr_x, curr_y

            if fingers[1] == 1 and fingers[2] == 1:
                length, img, lineInfo = detector.findDistance(8, 12, img)

                if length < 40:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    autopy.mouse.click()

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)