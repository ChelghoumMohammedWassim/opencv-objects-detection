import numpy as np
import cv2 as cv
import pyautogui
from time import time
from PIL import ImageGrab
import win32gui
from tools.windowCapture import WindowCapture
from tools.edgeFilter import EdgeFilter
from mss import mss
from vision import *

#target window name
windowName='Photos'

#get all windows names
WindowCapture.list_window_names()

#create capture instance
winCap=WindowCapture(windowName=None)

#get device resolution
w, h = pyautogui.size()
print("Screen Resolution: " + str(w) + 'x' + str(h))

#target window 
hwnd=win32gui.FindWindow(None,windowName)
hwnd=0x1b063a
#display fps 
def update_line(message):
    print(message, end='\r')

#read target image
needle_img = cv.imread('images/filtred_cabbage.jpg', cv.IMREAD_UNCHANGED)

# filter.init_control_gui()
filter=EdgeFilter()

filter.init_control_gui()

loop_time=time()
while True:
        #get target pixels
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        
        #take screenshot
        screenshot=ImageGrab.grab(bbox=(left, top, right, bottom))
    
        #convert to array and convert from RGB to BGR
        screenshot=np.array(screenshot)
        screenshot=cv.cvtColor(screenshot, cv.COLOR_RGB2BGR)
        
        #apply filter
        filtered_image=filter.apply_edge_filter(screenshot)
        
        #identify target positions
        # rectangles=getRectangles(filtered_image,needle_img,0.32)
        # click_points=getPoints(rectangles)
        
        # #draw information on screenshot
        # template=drawRectangles(rectangles,screenshot)
        # template=drawCrosshair(click_points,template)
        
        
        #small image
        # result_template=cv.resize(template,(0,0),fx=0.7,fy=0.7)
        result_filtered_image=cv.resize(filtered_image,(0,0),fx=0.6,fy=0.6)
        
        # cv.imshow('Computer Vision',result_template)
        cv.imshow('Computer Treatment',result_filtered_image)
            
        #show how much time tike 1 screenshot
        update_line(f"FPS: {1/(time()-loop_time)} Delay:{time()-loop_time}")
        loop_time=time()
        
        if cv.waitKey(1)==ord('*'):
            cv.destroyAllWindows()
            break


#first method PIL only position of target fps 20-19

    # #get target pixels
    # left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    
    # #take screenshot
    # screenshot=ImageGrab.grab(bbox=(left, top, right, bottom))
   
    # #convert to array and convert from RGB to BGR
    # screenshot=np.array(screenshot)
    # screenshot=cv.cvtColor(screenshot, cv.COLOR_RGB2BGR)

#second method mss only position of target fps (30-20) high ram and disk usage:
    # #get target pixels
    # left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    # monitor = {"top": 0, "left": 0, "width": w, "height": h}
    # #take screenshot
    # screenshot= mss().grab(monitor)
    # screenshot=np.array(screenshot)

#third method win32 best fps low precision fps: depend en screen resolution(1080p 20-19)
#screenshot= winCap.get_screenshot()


