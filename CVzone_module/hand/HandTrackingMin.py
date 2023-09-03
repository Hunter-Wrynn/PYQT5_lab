import cv2
import mediapipe as mp
import time
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils
cap=cv2.VideoCapture(0)

pTIme=0
cTime=0

# 加载视频动画
animation_path = "../video/cz.mp4"
animation_cap = cv2.VideoCapture(animation_path)
# 选择目标点位置
target_point = (230, 150)  # 修改为你希望插入动画的目标点位置

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        hand_landmarks = []
        for handLms in results.multi_hand_landmarks:
            hand_landmarks.append(handLms)

            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        if len(hand_landmarks) == 2:
            # 插入视频动画
            ret_anim, frame_anim = animation_cap.read()
            if ret_anim:
                frame_anim_resized = cv2.resize(frame_anim, (250, 300))  # 调整动画大小
                x, y = target_point
                img[y:y + frame_anim_resized.shape[0], x:x + frame_anim_resized.shape[1]] = frame_anim_resized

    cTime=time.time()
    fps=1/(cTime-pTIme)
    pTIme=cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,2555),3)
    cv2.imshow('img',img)
    cv2.waitKey(1)
