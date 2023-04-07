import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


def get_thresh(img):
    
    kernel5 = np.ones((5,5),np.uint8)
    kernel3 = np.ones((3,3),np.uint8)
    
    # Repeated Closing operation to remove text from the document.
    no_content = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel5, iterations= 3) 
    sheet_gray = cv2.cvtColor(no_content, cv2.COLOR_BGR2GRAY)
    sheet_gray = cv2.GaussianBlur(sheet_gray,(5,5),0)
    
    sobelx = cv2.Sobel(sheet_gray,cv2.CV_16S,1,0,ksize=5)
    sobely = cv2.Sobel(sheet_gray,cv2.CV_16S,0,1,ksize=5)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    sobel = cv2.addWeighted(src1=abs_sobelx,alpha=0.5,src2=abs_sobely,beta=0.5,gamma=0)
   
    _,sobel_thresh = cv2.threshold(sobel,sobel.max()*0.1,255,cv2.THRESH_BINARY) 
    #sobel_thresh = cv2.erode(sobel_thresh,kernel3,iterations=1)
    sobel_thresh = sobel_thresh.astype(np.uint8)

    return sobel_thresh



def draw_contour(img, contour):

    sheet_indicators = img
    #sheet_indicators = cv2.merge((img,img,img))
    if len(contour) > 0:
        cv2.drawContours(sheet_indicators,[contour],-1,(0,255,0),2)
    
    return sheet_indicators



def draw_lines(img):
    
    # Apply HoughLinesP method to 
    # to directly obtain line end points
    lines = cv2.HoughLinesP(
        img, # Input edge image
        1, # Distance resolution in pixels
        np.pi/180, # Angle resolution in radians
        threshold=100, # Min number of votes for valid line
        minLineLength=100, # Min allowed length of line
        maxLineGap=20 # Max allowed gap between line for joining them
        )                

    img = cv2.merge((img,img,img))    
    img = np.zeros(img.shape, np.uint8)
    
    # Define the extension length
    extension_length = 10
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            dx /= length
            dy /= length
            x1 -= dx * extension_length
            y1 -= dy * extension_length
            x2 += dx * extension_length
            y2 += dy * extension_length
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  
    return img



def get_biggest_contour(contours):
    biggest = np.array([])
    max_area = 5000
    for i in contours:
        area = cv2.contourArea(i)
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i,0.02*peri,True)
        if len(approx) == 4 and cv2.isContourConvex(approx) and area > max_area:
            biggest = approx
            max_area = area
            
    return [biggest, max_area]



def smooth_corners(buffer):
    smoothed_corners = []
    if len(buffer) > 0:
        smoothed_corners = np.mean(buffer, axis=0)
        smoothed_corners = np.round(smoothed_corners).astype(np.int32).reshape((-1, 1, 2))
    return smoothed_corners



def four_corners_sort(pts):
    """ Sort corners: top-left, bot-left, bot-right, top-right """
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)
    return np.array([pts[np.argmin(summ)],
                     pts[np.argmax(diff)],
                     pts[np.argmax(summ)],
                     pts[np.argmin(diff)]])



def contour_offset(cnt, offset):
    """ Offset contour, by 5px border """
    cnt += offset  
    # if value < 0 => replace it by 0
    cnt[cnt < 0] = 0
    return cnt



def warp_perspective(img, corners):
    if len(corners) > 0 and corners.shape == (4, 1, 2):
        corners = np.float32(np.reshape(corners, (4, 2)).squeeze())
        corners = four_corners_sort(corners)
        #corners = contour_offset(corners, (-20, -20))

        # Using Euclidean distance
        # Calculate maximum height (maximal length of vertical edges) and width
        height = max(np.linalg.norm(corners[0] - corners[1]),
                     np.linalg.norm(corners[2] - corners[3]))
        width = max(np.linalg.norm(corners[1] - corners[2]),
                     np.linalg.norm(corners[3] - corners[0]))
        window_points = np.array([[0, 0],[0, height],[width, height],[width, 0]], np.float32)
        if corners.dtype != np.float32:
            corners = corners.astype(np.float32)
    
    
        matrix = cv2.getPerspectiveTransform(corners, window_points)
        img = cv2.warpPerspective(img, matrix,(int(width),int(height)))
    return img








cap = cv2.VideoCapture(1)
cv2.namedWindow('Sheet Detect')
cv2.namedWindow('Scanned Document')

buffer_size = 10
corner_buffer = []
counter = 0

max_consecutive_misses = 10

while True:
    
    ret,frame = cap.read(0)
    thresh = get_thresh(frame)
    hough = draw_lines(thresh)
    
    contours, hierarchy = cv2.findContours(hough, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea,reverse=True)
    
    biggest, max_area = get_biggest_contour(contours)

    if len(biggest) > 0:
        # Add the corners to the buffer
        biggest = np.reshape(biggest, (4, 2)).squeeze()
        corner_buffer.append(biggest)
        # If the buffer is full, remove the oldest set of corners
        if len(corner_buffer) > buffer_size:
            corner_buffer.pop(0)
            
        counter = 0
    else:
        counter += 1
        if counter >= max_consecutive_misses:
            corner_buffer = []  # reset the buffer
            counter = 0  # reset the counter
            
        
    # Calculate the moving average of the corners over the buffer
    smoothed_corners = smooth_corners(corner_buffer)
        
    cv2.imshow('Scanned Document', warp_perspective(frame, smoothed_corners))
    cv2.imshow('Sheet Detect', draw_contour(frame, smoothed_corners))

    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()