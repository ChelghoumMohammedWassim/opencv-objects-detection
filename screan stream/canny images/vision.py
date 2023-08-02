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


def matchsPoints(template,target_img,patch_size=32):
    
    min_match_count = 3

    orb = cv.ORB_create(edgeThreshold=0, patchSize=patch_size)
    keypoints_needle, descriptors_needle = orb.detectAndCompute(target_img, None)
    orb2 = cv.ORB_create(edgeThreshold=0, patchSize=patch_size, nfeatures=2000)
    keypoints_haystack, descriptors_haystack = orb2.detectAndCompute(template, None)

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, 
            table_number=6,
            key_size=12,    
            multi_probe_level=1)

    search_params = dict(checks=50)

    try:
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors_needle, descriptors_haystack, k=2)
    except cv.error:
        return None, None, [], [], None

        # store all the good matches as per Lowe's ratio test.
    good = []
    points = []

    for pair in matches:
        if len(pair) == 2:
            if pair[0].distance < 0.7*pair[1].distance:
                good.append(pair[0])

    if len(good) > min_match_count:
        print('match %03d, kp %03d' % (len(good), len(keypoints_needle)))
        for match in good:
            points.append(keypoints_haystack[match.trainIdx].pt)
        #print(points)
        
    return keypoints_needle, keypoints_haystack, good, points

def centeroid(self, point_list):
    point_list = np.asarray(point_list, dtype=np.int32)
    length = point_list.shape[0]
    sum_x = np.sum(point_list[:, 0])
    sum_y = np.sum(point_list[:, 1])
    return [np.floor_divide(sum_x, length), np.floor_divide(sum_y, length)]