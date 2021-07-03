"""
This is the master file for Detection the Curved lanes.
Advanced Lane Finding Project
The goals / steps of this project are the following:
    1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    2. Apply a distortion correction to raw images.
    3. Use color transforms, gradients, etc., to create a thresholded binary image.
    4. Apply a perspective transform to rectify binary image ("birds-eye view").
    5. Detect lane pixels and fit to find the lane boundary.
    6. Determine the curvature of the lane and vehicle position with respect to center.
    7. Warp the detected lane boundaries back onto the original image.
    8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

"""
# -*- coding: utf-8 -*-
#importing some useful packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from moviepy.editor import VideoFileClip
import glob


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

def caliberate_camera():
    """
    This function is to be called only first to calibrate the camera and return
    the Camera Matrix and Distortion coefficients
    Returns
    -------
    Camera Matrix and Distortion coefficients: mtx, dist.
    """
    # prepare object points
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')
    
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            img_size = (img.shape[1], img.shape[0])
            # Do camera calibration given object points and image points
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
            dst = cv2.undistort(img, mtx, dist, None, mtx)
            write_name = 'output_images/Calibration_output/corners_found'+str(idx)+'.jpg'
            cv2.imwrite(write_name,dst)
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            ax1.imshow(img)
            ax1.set_title('Original Image', fontsize=30)
            ax2.imshow(dst)
            ax2.set_title('Undistorted Image', fontsize=30)
            cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    return mtx, dist

def undistort_image(img,mtx,dist):
    """
    This function takes in the raw image and the camera matrix and the Distortion
    coefficients calculated from the caliberate_camera() function, and undistorts
    the input image    
    
    Parameters
    ----------
    img : gray scale image.
    mtx : camera matrix from the caliberate_camera() function.
    dist : distortion coefficients from the caliberate_camera() function.

    Returns
    -------
    the undistorted gray scale image: undist.
    
    """
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def colour_gradient_transform(image, hue_thresh=(10, 30), sat_thresh=(150, 190), sx_thresh=(20, 80), sy_thresh=(20, 80), sAvg_thresh=(20, 80), sDir_thresh=(0, np.pi/2), kernel_size=3):
    """
    This function applies Sobel transform to find out gradients in the x direction,
    finds the absolute and scales them to uint8 range.Then this function applies 
    the HSV / HSL color space to detect the yellow and while lane lines under all 
    the lightinig conditions.Both the outputs are compared with a thereshold to 
    generate a binary image, and finally both are fused together to produce a single
    Binary image.
    
    Parameters
    ----------
    img : RGB Image
        input image.
    s_thresh : tuple --> higher and lower thresholds for detection of yellow lanes
        DESCRIPTION. The default is (150, 190).
    sx_thresh : tuple --> higher and lower thresholds for detection of left and right lanes
        DESCRIPTION. The default is (20, 80).

    Returns
    -------
    total_binary: the combined binary image from both Sobel Transform and Color Space transforms

    """
    img = np.copy(image)
    # Convert to different colour spaces
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #Equalize the histogram of the gray scale image the equalize the contrast / lighting
    #gray = cv2.equalizeHist(gray)
    """
    working on Sobel Gradients
    trying out different combinations.
    take sobel on x,y direction on a gray scale image
    Avg them and save as seperate img
    Also take Dir Grad on the Gray scale image
    """
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size) # Take the derivative in x
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size) # Take the derivative in y
    abs_sobel_x = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    abs_sobel_y = np.absolute(sobely) # Absolute y derivative to accentuate lines away from vertical
    scaled_sobel_x = np.uint8(255*abs_sobel_x/np.max(abs_sobel_x))
    scaled_sobel_y = np.uint8(255*abs_sobel_y/np.max(abs_sobel_y))
    abs_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_abs_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    dir_grad = np.arctan2(abs_sobel_y,abs_sobel_x)

    sxbinary = np.zeros_like(scaled_sobel_x)
    sxbinary[(scaled_sobel_x >= sx_thresh[0]) & (scaled_sobel_x <= sx_thresh[1])] = 1
    sybinary = np.zeros_like(scaled_sobel_y)
    sybinary[(scaled_sobel_y >= sy_thresh[0]) & (scaled_sobel_y <= sy_thresh[1])] = 1
    sAvgbinary = np.zeros_like(scaled_abs_sobel)
    sAvgbinary[(scaled_abs_sobel >= sAvg_thresh[0]) & (scaled_abs_sobel <= sAvg_thresh[1])] = 1
    sDirbinary = np.zeros_like(dir_grad)
    sDirbinary[(dir_grad >= sDir_thresh[0]) & (dir_grad <= sDir_thresh[1])] = 1
    sGradBinary = np.zeros_like(gray)
    sGradBinary[((sAvgbinary == 1) & (sDirbinary == 1))] = 1
    """
    Now work on Color Channels
    """
    # Threshold color channel
    sat_binary = np.zeros_like(hls[:,:,2])
    hue_binary = np.zeros_like(hls[:,:,2])
    color_binary = np.zeros_like(hls[:,:,2])
    total_binary = np.zeros_like(hls[:,:,2])
    sat_binary[(hls[:,:,2] >= sat_thresh[0]) & (hls[:,:,2] <= sat_thresh[1])] = 1
    hue_binary[(hls[:,:,0] >= hue_thresh[0]) & (hls[:,:,0] <= hue_thresh[1])] = 1
    color_binary[(sat_binary == 1) | (hue_binary == 1)] = 1
    total_binary[((color_binary == 1) | (sGradBinary == 1))] = 1
    return total_binary

def perspective_transform(img):
    """
    This class takes in the Gradient and Color Space thresholded binary image,
    and perspective transforms the Binary image to halp make view the lanes to be
    viewed from the top (as like in Bird's eye view). We take 4 source points that 
    are assumed to form a Quadrilateral and 4 destination points in the Warped image
    section. This class maps the points from Source points to the Destiation points.
    
    Parameters
    ----------
    img : Gradient and Color Space thresholded Binary image

    Returns
    -------
    warped : TYPE: Binary
        DESCRIPTION: Top Down perspective transformed image.
    M : TYPE: Matrix
        DESCRIPTION. Matrix used to Transform the perspective. This is to be used 
        Later on to invert the perspective.

    """
    # Use the OpenCV undistort() function to remove distortion
    offset = 400 # offset for dst points
    midpoint = img.shape[1] / 2
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])
    #create source points
    """
    src_point_1 = [590,450]
    src_point_2 = [690,450]
    src_point_3 = [1120,720]
    src_point_4 = [190,720]
    """
    src_point_1 = [590,450]
    src_point_2 = [690,450]
    src_point_3 = [1210,720]
    src_point_4 = [290,720]
    # For source points I'm grabbing the outer four detected corners
    src = np.float32([src_point_1,src_point_2,src_point_3,src_point_4])
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    #create desctination points
    dst_point_1 = [(midpoint - offset),0]
    dst_point_2 = [(midpoint + offset),0]
    dst_point_3 = [(midpoint + offset),img_size[1]]
    dst_point_4 = [(midpoint - offset),img_size[1]]
    dst = np.float32([dst_point_1, dst_point_2, dst_point_3, dst_point_4])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    #get the inverse Transform also..
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    f,(ax1,ax2)=plt.subplots(1,2,figsize=(20,10))
    ax1.imshow(img, cmap='gray')
    ax1.plot(590,450,'.')
    ax1.plot(690,450,'.')
    ax1.plot(1210,720,'.')
    ax1.plot(290,720,'.')
    ax2.imshow(warped, cmap='gray')
    # Return the resulting image and matrix
    return warped, M, Minv

def find_lane_pixels(binary_warped):
    """
    This class is called the first time when there is no lane already being
    detected. First we detect the lane points using histogram at the bottom of
    the image and use the same to set the Window height and width. This window 
    slides from Bottom to top and for each slide within each of the left and the
    right windows we detect the pixels that corresponds to the lane lines. These 
    Pixels are collected and the x and y values of these pixels are output from 
    this class.

    Parameters
    ----------
    binary_warped : binary image
        DESCRIPTION: input warped image (showing a top down view of the lanes)

    Returns
    -------
    leftx : TYPE
        DESCRIPTION.
    lefty : TYPE
        DESCRIPTION.
    rightx : TYPE
        DESCRIPTION.
    righty : TYPE
        DESCRIPTION.
    out_img : TYPE
        DESCRIPTION.

    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 30

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    
    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    """
    

    Parameters
    ----------
    binary_warped : TYPE
        DESCRIPTION.

    Returns
    -------
    out_img : TYPE
        DESCRIPTION.

    """
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    
    #Check if the lanes are found or not found
    if ((len(leftx) > 0) & (len(lefty) > 0)):
        left_lanes_found = True
    else:
        left_lanes_found = False
    if ((len(rightx) > 0) & (len(righty) > 0)):
        right_lanes_found = True
    else:
        right_lanes_found = False

    return left_fitx, right_fitx, left_lanes_found, right_lanes_found, left_fit, right_fit

def fit_poly_new(img_shape, leftx, lefty, rightx, righty):
    """
    

    Parameters
    ----------
    img_shape : TYPE
        DESCRIPTION.
    leftx : TYPE
        DESCRIPTION.
    lefty : TYPE
        DESCRIPTION.
    rightx : TYPE
        DESCRIPTION.
    righty : TYPE
        DESCRIPTION.

    Returns
    -------
    left_fit_new : TYPE
        DESCRIPTION.
    left_fit_new : TYPE
        DESCRIPTION.
    left_fitx : TYPE
        DESCRIPTION.
    right_fitx : TYPE
        DESCRIPTION.
    ploty : TYPE
        DESCRIPTION.

    """
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit_new = np.polyfit(lefty, leftx, 2)
    right_fit_new = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    try:
        left_fitx = left_fit_new[0]*ploty**2 + left_fit_new[1]*ploty + left_fit_new[2]
        right_fitx = right_fit_new[0]*ploty**2 + right_fit_new[1]*ploty + right_fit_new[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    
    return left_fit_new, left_fit_new, left_fitx, right_fitx, ploty

def search_around_poly(binary_warped,Left_Fit,Right_Fit):
    """
    

    Parameters
    ----------
    binary_warped : TYPE
        DESCRIPTION.
    Left_Fit : TYPE
        DESCRIPTION.
    Right_Fit : TYPE
        DESCRIPTION.

    Returns
    -------
    result : TYPE
        DESCRIPTION.
    left_fitx : TYPE
        DESCRIPTION.
    right_fitx : TYPE
        DESCRIPTION.
    left_lanes_found : TYPE
        DESCRIPTION.
    right_lanes_found : TYPE
        DESCRIPTION.
    left_fit_new : TYPE
        DESCRIPTION.
    right_fit_new : TYPE
        DESCRIPTION.

    """
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_fit = Left_Fit
    right_fit = Right_Fit
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    
    #left_lane_inds = []
    #right_lane_inds = []
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin))).nonzero()[0]
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin))).nonzero()[0]
                                                           
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fit_new, right_fit_new, left_fitx, right_fitx, ploty = fit_poly_new(binary_warped.shape, leftx, lefty, rightx, righty)
    
    #Check if the lanes are found or not found
    if ((len(leftx) > 0) & (len(lefty) > 0)):
        left_lanes_found = True
    else:
        left_lanes_found = False
    if ((len(rightx) > 0) & (len(righty) > 0)):
        right_lanes_found = True
    else:
        right_lanes_found = False
    
    """
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    """
    
    return left_fitx, right_fitx, left_lanes_found, right_lanes_found, left_fit_new, right_fit_new

def image_reMap(binary_warped,left_fitx,right_fitx,Minv,undist):
    """
    

    Parameters
    ----------
    warped : TYPE
        DESCRIPTION.
    left_fitx : TYPE
        DESCRIPTION.
    right_fitx : TYPE
        DESCRIPTION.
    ploty : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    margin = 50
    #left_fitx = np.concatenate(left_fitx)
    #right_fitx = np.concatenate(right_fitx)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,0, 255))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (255,0, 0))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(window_img, Minv, (window_img.shape[1], window_img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    plt.imshow(result)
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))*255
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (binary_warped.shape[1], binary_warped.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    plt.imshow(result)
    """
    return result

def generate_data_real(ym_per_pix, xm_per_pix,image):
    """
    

    Parameters
    ----------
    ym_per_pix : TYPE
        DESCRIPTION.
    xm_per_pix : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    left_fit_cr : TYPE
        DESCRIPTION.
    right_fit_cr : TYPE
        DESCRIPTION.

    """
    leftx = ObjLeft_lane.recent_xfitted
    rightx = ObjRight_lane.recent_xfitted
    
    ploty = np.linspace(0, image[:,:,0].shape[0]-1, image[:,:,0].shape[0])# to cover same y-range as image

    leftx = np.transpose(np.array(leftx[::-1])) * xm_per_pix  # Reverse to match top-to-bottom in y
    rightx = np.transpose(np.array(rightx[::-1])) * xm_per_pix  # Reverse to match top-to-bottom in y

    # Fit a second order polynomial to pixel positions in each fake lane line
    ##### TO-DO: Fit new polynomials to x,y in world space #####
    ##### Utilize `ym_per_pix` & `xm_per_pix` here #####
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx, 2)
    
    return ploty * ym_per_pix, left_fit_cr, right_fit_cr

def measure_curvature_real(image):
    """
    

    Returns
    -------
    left_curverad : TYPE
        DESCRIPTION.
    right_curverad : TYPE
        DESCRIPTION.

    """
    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    ploty, left_fit_cr, right_fit_cr = generate_data_real(ym_per_pix, xm_per_pix,image)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = np.sqrt((np.square((2*left_fit_cr[0]*y_eval) + left_fit_cr[1]) + 1)**3)/(2*left_fit_cr[0])
    right_curverad = np.sqrt((np.square((2*right_fit_cr[0]*y_eval) + right_fit_cr[1]) + 1)**3)/(2*right_fit_cr[0])
    
    overall_radius = np.abs((left_curverad + right_curverad) * 0.5)
    
    return left_curverad, right_curverad, overall_radius

def find_distance_from_lanecentre(xFitLeft, xFitRight, ym_per_pix, xm_per_pix, image):
    """
    

    Parameters
    ----------
    left_curverad : TYPE
        DESCRIPTION.
    right_curverad : TYPE
        DESCRIPTION.
    image : TYPE
        DESCRIPTION.

    Returns
    -------
    Offset : TYPE
        DESCRIPTION.

    """
    """
    centre = np.abs(left_curverad - right_curverad) * 0.5
    image_centre = np.int(image[:,:,0].shape[0]//2) * xm_per_pix
    Offset_raw = centre - image_centre
    """
    xFitLeft = np.concatenate(xFitLeft)
    xFitRight = np.concatenate(xFitRight)
    xFitLeft_Bottom = xFitLeft[len(xFitLeft) - 1]
    xFitRight_Bottom = xFitRight[len(xFitRight) - 1]
    centre = xFitRight_Bottom - xFitLeft_Bottom
    image_centre = image[:,:,0].shape[1]/2
    Offset = (image_centre - centre) * xm_per_pix
    return Offset

def process_image(image):
    """
    

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.

    Returns
    -------
    projected_image : TYPE
        DESCRIPTION.

    """
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image where lines are drawn on lanes)
    # convert to gray scale
    undist_img = undistort_image(image,mtx,dist)
    #plt.imshow(undist_img)
    binary_image = colour_gradient_transform(undist_img,hue_thresholds, saturation_thresholds,xGrad_Thresholds,yGrad_Thresholds,AvgGrad_Threshlds,DirGrad_Thresholds,kernel_size)
    #plt.imshow(binary_image,cmap = 'gray')
    #pERSPECTIVE tRANSFORM THE IMAGE TO EXTRACT THE LEFT AND RIGHT LANE LINES
    warped_image, persp_matrix, inv_persp_matrix = perspective_transform(binary_image)
    #warped_image_color, persp_matrix_c, inv_persp_matrix_c = perspective_transform(undist_img)
    #FIT THE POLYNOMIAL AND SEARCH FOR LANE LINES WITHIN THE POLYNOMIAL BOUNDARY IN CONSECUTIVE FRAMES
    if ((ObjLeft_lane.detected == False) | (ObjRight_lane.detected == False)):
        left_fitx, right_fitx, left_lanes_found, right_lanes_found, left_fit_poly, right_fit_poly = fit_polynomial(warped_image)
        ObjLeft_lane.recent_xfitted.append(left_fitx)
        ObjLeft_lane.detected = left_lanes_found
        ObjLeft_lane.current_fit = left_fit_poly
        ObjRight_lane.recent_xfitted.append(right_fitx)
        ObjRight_lane.detected = right_lanes_found
        ObjRight_lane.current_fit = right_fit_poly
    else:
        left_fitx, right_fitx, left_lanes_found, right_lanes_found, left_fit_poly_new, right_fit_poly_new = search_around_poly(warped_image,ObjLeft_lane.current_fit,ObjRight_lane.current_fit)
        ObjLeft_lane.recent_xfitted.append(left_fitx)
        ObjLeft_lane.detected = left_lanes_found
        ObjLeft_lane.current_fit = left_fit_poly_new
        ObjRight_lane.recent_xfitted.append(right_fitx)
        ObjRight_lane.detected = right_lanes_found
        ObjRight_lane.current_fit = right_fit_poly_new
    #Project back the Polynomial on to the Images
    #undist_gray = cv2.cvtColor(undist_img, cv2.COLOR_RGB2GRAY)
    projected_image = image_reMap(warped_image,left_fitx,right_fitx,inv_persp_matrix,undist_img)
    #FIND THE RADIUS OF CURVATURE
    ObjLeft_lane.radius_of_curvature, ObjRight_lane.radius_of_curvature, overall_radius = measure_curvature_real(projected_image)
    distance_from_centre = find_distance_from_lanecentre(ObjLeft_lane.recent_xfitted, ObjRight_lane.recent_xfitted, ym_per_pix, xm_per_pix, projected_image)
    #embed the text in the image
    text1 = "The Radius of Curvature is: " + str(overall_radius) + "meters"
    if (distance_from_centre < 0):
        text2 = "the vehicle deviates " + str(np.abs(distance_from_centre)) + "meters to the left"
    else:
        text2 = "the vehicle deviates " + str(np.abs(distance_from_centre)) + "meters to the right"
    cv2.putText(projected_image,text1,(10,100), cv2.FONT_HERSHEY_COMPLEX, 2,(0,255,0),2,cv2.LINE_AA)
    cv2.putText(projected_image,text2,(10,200), cv2.FONT_HERSHEY_COMPLEX, 2,(0,255,0),2,cv2.LINE_AA)
    return projected_image

#the first step --> caliberate the camera
#mtx, dist = caliberate_camera()
mtx = np.array([[1157.78,0,667.114],[0,1152.82,386.125],[0,0,1]])
dist = np.array([[-0.246885,-0.0237315,-0.00109831,0.00035107,-0.00259868]])
#Define some constants..
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
hue_thresholds = [20,28]
saturation_thresholds = [100,150]
xGrad_Thresholds = [100,200]
yGrad_Thresholds = [100,200]
AvgGrad_Threshlds = [50,200]
DirGrad_Thresholds = [0.7,1.5]
kernel_size = 3
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
ObjLeft_lane = Line()
ObjRight_lane = Line()
Video_op_path = "C:\\Users\\bra6cob\\Documents\\GitHub\\CarND-Advanced-Lane-Lines\\test_video_outputs\\project_video_output.mp4"
clip = VideoFileClip("C:\\Users\\bra6cob\\Documents\\GitHub\\CarND-Advanced-Lane-Lines\\project_video.mp4")
Raw_clip = clip.fl_image(process_image)
Raw_clip.write_videofile(Video_op_path, audio=False)
"""
clip = clip.subclip(0, 1)
frames = clip.iter_frames()
framecounter = 0
for value in frames:
    image = plt.imshow(value)
    img = image._A
    framecounter = framecounter + 1
    projected_image = process_image(img)
    print(framecounter)
"""