
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pylab', 'notebook')
import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, lines_parameters):
    slope_intercept = lines_parameters
    y1 = image.shape[0]
    y2 = int (y1*(3/5))
    x1 = int ((y1-intercept)/slope)
    x2 = int ((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])
    
def average_slope_intercept (image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope_intercept))
        else:
            right_fit.append(slope_intercept)
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_lines = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array ([left_lines, right_line])
        

def canny(lane_images):
#for converting the image to grayscale (cv2.cvtColor) is the function for changing the image to gray
    gray = cv2.cvtColor(lane_images, cv2.COLOR_RGB2GRAY)  
#adding a gaussian blur to the gray image
    blur = cv2.GaussianBlur(gray, (5,5), 0) # Where'0' is the deviation 
#Using Canny function to detect the edges
    canny = cv2.Canny (blur, 80, 160) # 50 is the lower threshold and 150 is the upper threshold
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (10, 250, 100), 7)
    return line_image
    

def region_of_interest(image):
#height is set for the y-axis
    height = image.shape[0]
#dimension of the polygon (left XY), (Right XY), (Apex XY)
    polygon = np.array ([[(136, height), (800, height), (490, 228)]])
    mask = np.zeros_like(image)
    cv2.fillPoly (mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
    
image = cv2.imread('solidWhiteRight.jpg') #This command is used to load image 
lane_images = np.copy(image) #For making the copy of the image. 
canny_image = canny(lane_images)
cropped_image = region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average_slope_intercept(lane_images, lines)
line_image = display_lines(lane_images, averaged_lines)
combo = cv2.addWeighted(lane_images, 0.8, line_image, 1, 1)
# Function for displaying the image 
cv2.imshow("result", combo)
cv2.waitKey(0)

cap = cv2.VideoCapture("solidWhiteRight.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_images = display_lines(frame, averaged_lines)
    combo = cv2.addWeighted(frame, 0.8, line_images, 1, 1)
# Function for displaying the image 
    cv2.imshow("result", combo)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv.destroyAllwindows()
    

