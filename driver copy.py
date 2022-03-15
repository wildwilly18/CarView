import pygetwindow
import time
import os
import pyautogui
from PIL import ImageGrab, ImageShow, Image


z = pygetwindow.getWindowsWithTitle('Forza Horizon 5')[0]
time.sleep(1)
z.moveTo(0, 0)
z.resizeTo(1200, 780)

def talkAndSleep(wait_time):
    txt = "Seconds "
    while(wait_time>-1):
        print(wait_time)
        time.sleep(1)
        wait_time -= 1

def savePictures(numPictures, path):
    ii = 0
    while True:
        img = pyautogui.screenshot(region=(z.topleft.x+7, z.topleft.y, z.bottomright.x-14, z.bottomright.y-12))
        img_name = path + str(ii) + ".png"
        img.save(img_name)
        print("Saved " + img_name)
        talkAndSleep(5)
        if(ii > numPictures):
            break
        ii += 1

savePictures(50, "C:\RoadDetect\Pictures\StructuredRoad\sroad_")
    