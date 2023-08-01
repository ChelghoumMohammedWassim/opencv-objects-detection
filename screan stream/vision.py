import cv2 as cv
import numpy as np


def getRectangles(template, target_image, min_rate=0.5):
    
    result = cv.matchTemplate(template, target_image, cv.TM_CCOEFF_NORMED)

    locations=np.where(result>min_rate)

    if locations:

        locations=list(zip(*locations[::-1]))
        
        w=target_image.shape[1]
        h=target_image.shape[0]
        
        rectangles=[]
        
        for loc in locations:
            rectangles.append([loc[0],loc[1],w,h])
            rectangles.append([loc[0],loc[1],w,h])
    
    rectangles, _ =cv.groupRectangles(rectangles,1,0.5)
    
    return rectangles
   
        
def getPoints(rectangles):
    
    points=[]
        
    for (x,y,w,h) in rectangles:
        center=(x+(w/2),y+(h/2))
        points.append(center)
    
    return points        
    
    
def drawRectangles(rectangles,template):
    
    for (x,y,w,h) in rectangles:
        start=(x,y)
        end=(x+w,y+h)   
        cv.rectangle(template,start,end,color=(0,255,0),thickness=1,lineType=cv.LINE_4)
        
    return template


def drawCrosshair(points,template):
    
    for point in points:
        cv.drawMarker(template,position= (int(point[0]),int(point[1])), color=(0,0,255), markerType=cv.MARKER_TILTED_CROSS)
        
    return template
