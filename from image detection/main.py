import cv2 as cv
import numpy as np


def getPoints(template,target):
    
    points=[]
    
    result = cv.matchTemplate(haystack_img, needle_img, cv.TM_SQDIFF_NORMED)


    locations=np.where(result<0.171)

    if locations:

        locations=list(zip(*locations[::-1]))
        
        w=needle_img.shape[1]
        h=needle_img.shape[0]
        
        rectangles=[]
        
        for loc in locations:
            rectangles.append([loc[0],loc[1],w,h])
            rectangles.append([loc[0],loc[1],w,h])
        
        
        rectangles, _ =cv.groupRectangles(rectangles,1,0.5)
        
        
        print('\nfounded matches :')
        for (x,y,w,h) in rectangles:
            start=(x,y)
            end=(x+w,y+h)
            center=(x+(w/2),y+(h/2))
            points.append(center)
            
            print('\tstart location '+str(start)+' end location '+str(end)+' center '+str(center))
            
            cv.rectangle(haystack_img,start,end,color=(0,255,0),thickness=1,lineType=cv.LINE_4)
            cv.drawMarker(haystack_img,position= (int(center[0]),int(center[1])), color=(0,0,255), markerType=cv.MARKER_TILTED_CROSS)

        cv.imshow('result',haystack_img)
        cv.waitKey()
        
        return points
    
    else:
        print('not found')
        return None
    

haystack_img = cv.imread('images/albion_farm.jpg', cv.IMREAD_UNCHANGED)
needle_img = cv.imread('images/albion_cabbage.jpg', cv.IMREAD_UNCHANGED)

targetPoints=getPoints(template= haystack_img, target= needle_img)
print('\npoint to click :')
print(targetPoints)