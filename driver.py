import pygetwindow
import time
import os
import pyautogui
from PIL import ImageGrab, ImageShow, Image
import cv2
import numpy as np
import mss
from lane_find import *


z = pygetwindow.getWindowsWithTitle('Forza Horizon 5')[0]
time.sleep(1)
z.moveTo(0, 0)
z.resizeTo(1200, 780)


winname = 'Display'

cv2.namedWindow(winname)
cv2.moveWindow(winname, -1600, 0)

with mss.mss() as sct:
    #Screen part to cap
    forzaWindow = {"top": z.topleft.y, "left":z.topleft.x+7, "width":z.bottomright.x-14, "height":z.bottomright.y-12}
    while True:
        #While the function is running grab
        img = sct.grab(forzaWindow)
        img = np.array(img)

        x,y,lane1,lane2 = lane_find(img)
        cv2.line(img, (int(lane1[0]),int(lane1[1])), (int(lane1[2]),int(lane1[3])), (255,150,0),thickness = 10)
        cv2.line(img, (int(lane2[0]),int(lane2[1])), (int(lane2[2]),int(lane2[3])), (150,255,0),thickness = 10)
        cv2.circle(img, (int(x),int(y)), 7, (255,0,0), -1)

        cv2.imshow(winname, img) 
        cv2.waitKey(33)

    #img = pyautogui.screenshot(region=(z.topleft.x+7, z.topleft.y, z.bottomright.x-14, z.bottomright.y-12))
    #img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #cv2.imshow(winname, img) 
    #cv2.waitKey() 