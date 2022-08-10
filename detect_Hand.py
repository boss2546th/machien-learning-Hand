import cv2
import mediapipe as mp
import time
import pyfirmata as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#bot = fm.Arduino("COM4")
#bot.digital[11].mode = fm.OUTPUT

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex ,self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return lmlist

relate_xy  = [[],[]]

def main():
    global bot
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    width = 0
    zoom = 0
    show_pos = (0,0)
    percent = 0
    state = 'without'
    model = "(1.47092751*zoom) - (6.82653869)"
    max_width_used = 230
    clr_state = (0,0,0)
    color = 0
    global relate_xy


    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        opt = ''
        if len(lmlist) != 0:
            ID = [4,8]
            line = []
            width = (((lmlist[4][1]-lmlist[8][1])**2 )+((lmlist[4][2]-lmlist[8][2])**2))**0.5
            zoom = (((lmlist[0][1]-lmlist[5][1])**2 )+((lmlist[0][2]-lmlist[5][2])**2))**0.5

            max_width = eval(model)

            max_width_used = max_width if state == "with" else 230

            for i in ID:
                opt += " " + str(lmlist[i])
                cv2.circle(img, (lmlist[i][1], lmlist[i][2]), 10, (0, 0, 255), cv2.FILLED)
                line.append((lmlist[i][1], lmlist[i][2]))
            percent = (width/max_width_used)*100
            color = (255/100)*percent
            cv2.line(img, line[0], line[1], (0, int(color),0), 5)
            print(opt)
            show_pos = (int(line[1][0] + 25), int(line[1][1] + 25))

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        clr_state = (0, 255, 0) if state == "with" else (255, 255, 255)
        cv2.putText(img, "fps : "+str(int(fps)), (7, 470), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.putText(img, "width "+str(int(percent))+"%", show_pos, cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.putText(img, str(int(zoom)), (570, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        cv2.putText(img, "zoom", (550, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
        cv2.line(img, (580, 100), (620, 100), (0, 0, 0), 5)
        cv2.line(img, (600, 400), (600, 100), (0, 0, 0), 5)
        cv2.line(img, (600,400), (600,int(400-(zoom/250)*300)), (0, 255,0), 20)
        cv2.line(img, (580, 410), (620, 410), (0, 0, 0), 5)
        cv2.putText(img, state+" model machine learning", (55, 50), cv2.FONT_HERSHEY_PLAIN, 2, clr_state, 2)
        cv2.putText(img, model, (200, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        cv2.imshow("Image", img)
        relate_xy[0].append(int(zoom))
        relate_xy[1].append(int(width))

        #bot.digital[11].write(int((255/100)*percent))

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('m'):
            state = 'with'
        elif k == ord('n'):
            state = 'without'



def plot():
    global relate_xy
    x = np.array(relate_xy[0])
    y = np.array(relate_xy[1])
    print(relate_xy)
    plt.scatter(x,y)
    plt.xlabel("ZOOM")
    plt.ylabel("MAX DISTANT")
    plt.show()
    print(len(x))



if __name__ == "__main__":
    main()
    #plot()