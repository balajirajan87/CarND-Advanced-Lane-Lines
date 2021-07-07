**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./camera_cal/calibration10.jpg "Original Chessboard image"
[image1]: ./output_images/Calibration_output/corners_found1.jpg "Undistorted chessboard image"
[image2]: ./output_images/original_distorted_image.jpg "Original Distorted Image"
[image3]: ./output_images/undistorted_image.jpg "Undistorted Image"
[image4]: ./output_images/sobel_binary_x_dir.jpg "Sobel_x_direction image"
[image5]: ./output_images/sobel_binary_y_dir.jpg "Sobel_y_direction image"
[image6]: ./output_images/scaled_sobel_xy.jpg "overall scaled sobel image"
[image7]: ./output_images/dir_grad.jpg "Direction Gradient sobel image"
[image8]: ./output_images/sAvgBinary_image.jpg "thresholded average sobel image"
[image9]: ./output_images/sDirBinary_image.jpg "thresholded Direction sobel image"
[image10]: ./output_images/sGradBinary_image.jpg "combined threshold binary image"
[image11]: ./output_images/Hue_binary.jpg "Hue thresholded binary image"
[image12]: ./output_images/Sat_binary.jpg "Saturation thresholded binary image"
[image13]: ./output_images/color_binary.jpg "Final Color combined thresholded binary image"
[image14]: ./output_images/total_binary_image.jpg "Final Color and Gradient combined thresholded binary image"
[image15]: ./output_images/warped_binary_image.jpg "BirdsEye View warped binary image"
[image16]: ./output_images/Left_right_lanes_detected.jpg "Left and Right Lanes Detected"
[image17]: ./output_images/lane_lines.jpg "Lane Lines mapped"
[image18]: ./output_images/lane_lines_with_polygon.jpg "Lane lines with polygon maped"
[image19]: ./output_images/lanes_remaped_undistorted.jpg "Lane lines with polygon transformed onto undist image"
[image20]: ./output_images/final_image_withData.jpg "final image with Radius of Curvature data"
[video1]: ./test_video_outputs/project_video_output.mp4 "final video output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Below is the Detailed explanation of each part of the Code. 

---

### Definition of class Line(), and its Initialization.

First i define two classes names Line() and Frame(), and define its __init__() method which is used to initialize the Variables to a default initial value. The code for this step is located in the lines 27 through 55 of the main file called "Advanced_Lane_Detect.py".

### Definition of some Global parameters

Next i define a set of Global parameters / variables like : ObjLeft_lane, framecounter, ym_per_pix, ym_per_pix, hue_thresholds, saturation_thresholds, etc that are used within the functions called from the main video processing pipeline.
this you can find from the lines 57 through 85, in the file: "Advanced_Lane_Detect.py".

### Camera Calibration

This is the first function / the process that is being executed.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. 
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  
Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. 
using cv2.findChessboardCorners() function i find the chessboard corners and append the same to imgpoints, whenever the corner detection is successful. 
`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration matrix and distortion coefficients using the `cv2.calibrateCamera()` function.  
I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Original Image
![alt text][image0]
Undistorted Image
![alt text][image1]

### Pipeline (single images)

This Pipeline is used for processing each and every frame. Each frame correspond to a single image.

#### 1. Distortion correction to the camera frames.

Using the Camera matrix and Distortion Coefficients calculated from the Camera caliberation step, i apply cv2.undistort() function to each of the input image as shown below:
![alt text][image2]
![alt text][image3]

This process of undistorting each and every recieved frames is being taken care by undistort_image() function being called the first in the image processing pipeline.

#### 2. Color and Gradient Transforms to the original images

I have used a combination of average sobel gradients in both x and y direction, Direction Gradients, and Color Gradients to create a final Binary image that represents Lane Lines.
below are the steps followed and the reason for chosing the method.

First i convert the undistorted image to a Gray scale image using : cv2.cvtColor(img, cv2.COLOR_RGB2GRAY). Then with the Gray scale image i take the sobel gradients in both x and y directions as shown below
![alt text][image4]
![alt text][image5]
With the above two images we take the average of both of them using: "np.sqrt(np.square(sobelx) + np.square(sobely))", and the image obtained is as below:
![alt text][image6]
Next we take the Direction Gradient. i.e, the angle information between x and y gradients since our lanes are more closely associated with angle of pi/2. the output of Direction Gradient is as below:
![alt text][image7]
Next we thrshold the average sobel image using the threshold tuple: [50,200], and similarly we threshold the Direction Gradient image using the tuple: [0.7,1.5]. Please refer to the below images that describe the process
![alt text][image8]
![alt text][image9]
Next we combine both the above mentioned threshold images to a single threshold image bit by bitwise AND of both the images and the same is displayed below:
![alt text][image10]
Next we work on Colour Trasformations.
We extract Hue and Saturation spaces, and threshold those transformed spaces using the tuples: [20,28] for Hue space, and [100,150] for Saturation space. please refer the below images for the output thresholded images.
![alt text][image11]
![alt text][image12]
Next we combine the above generated binary images to a single color thresholded binary image by bitwise OR of both the above images. the resultant image represents color transformed binary image as shown below:
![alt text][image13]
now with the Color Transformed Binary image and Gradient Transformed Binary images as explained above, we generate a single Binary image by Bitwise OR of the Color Threshold Image and Gradient Threshold Image.
![alt text][image14]

#### 3. Perspective Transformation to the thresholded Binary Image to generate a BirdEye View of the binary image (to identify the lane lines)

This function is to transform the image to a BirdsEye (i.e, top down view) view image to detect and map the lane lines.
The code for my perspective transform includes a function called `perspective_transform()`, which appears in lines 245 through 309 in the main python file `Advanced_Lane_Detect.py`
The `perspective_transform()` function takes as inputs the thresholded binary image (`img`).
The source (src[]) and destination points (dst[]) to transform the image are created manually within the section of the code.
I chose to hardcode the source and destination points in the following manner:

```python
	src_point_1 = [590,450]
    src_point_2 = [690,450]
    src_point_3 = [1210,720]
    src_point_4 = [290,720]
    
    src = np.float32([src_point_1,src_point_2,src_point_3,src_point_4])
	
	dst_point_1 = [(midpoint - offset),0]
    dst_point_2 = [(midpoint + offset),0]
    dst_point_3 = [(midpoint + offset),img_size[1]]
    dst_point_4 = [(midpoint - offset),img_size[1]]
	
    dst = np.float32([dst_point_1, dst_point_2, dst_point_3, dst_point_4])
```

The midpoint is the bottom centre of the image and is known from : midpoint = img.shape[1] / 2
Offset value is chosen to be: 400

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 590, 450      | 240, 0        | 
| 690, 450      | 1040, 0      	|
| 1210, 720     | 1040, 720     |
| 290, 720      | 240, 7200     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. please see below

![alt text][image15]

#### 4. Lane Detection

The First Step in Lane Detection is to take the Warped Binary Image and find the x position of the image where there is a heavy concentration of While Pixels (with a value of 1) this is done by taking the histogram of the image.
The midpoint , Leftx_base and Rightx_base that corresponds to the starting point of the left and the right lanes are found out, and these two values are assigned to Leftx_current and Rightx_Current which would be used in the next cycle to find the average and move the window.
with the input binary image nonzero pixel indices in both x and y direction are found out and stored in an array.
Here we work with a sliding window method where we start from the bottom, and move up by finding the average of the pixel positions. for this a variable window with value 9 is being defined and this defines the window height and width.
This is done as below:

```python

	win_y_low = binary_warped.shape[0] - (window+1)*window_height
	win_y_high = binary_warped.shape[0] - window*window_height
	win_xleft_low = leftx_current - margin
	win_xleft_high = leftx_current + margin
	win_xright_low = rightx_current - margin
	win_xright_high = rightx_current + margin
	
```
with the above defined values: rectangles are drawn on the binary image as like below:

```python

	# Draw the windows on the visualization image
	cv2.rectangle(out_img,(win_xleft_low,win_y_low),
	(win_xleft_high,win_y_high),(0,255,0), 2) 
	cv2.rectangle(out_img,(win_xright_low,win_y_low),
	(win_xright_high,win_y_high),(0,255,0), 2)
	
```

Now with the window information and with the nonzerox and nonzeroy information, only the pixel indices that fall within window are captured and are apended to the array: left_lane_inds[] and right_lane_inds[]
The Left and Right Lane indices are passed to the Nonzerox and nonzeroy arrays and x and y positions of the left and right lanes are obtained.
Now these x and y values are now used to get the polynomial coefficients using np.polyfit() function. we obtain the polynomial coefficients for both the left and the right lanes. 
this along with ploty which is essentially a linspace mapping of the y coordinates of the image, we obtain the left and the right xFit values. these are plotted down to an image as below

![alt text][image16]

If the number of pixels found in the left and in the right lanes are greater than a threshold, then we say lanes are found.
the left_fitx, right_fitx, left_fit_poly, right_ft_poly and the lane found informations are stored in the global object variables by calling the Lane() classes. 
the detected Left and Right Lanes are being used in the next and subsequent frames to search for the lane pixels around the already detected pixels.
This is achieved using search_around_poly() function with binary warped image, left_fit (polynomial) and right_fit (polynomial) values as the arguments.
In this function first with the previous values of Left and the Right Fits, we get the corresponding xFit values. The calculated xFit values are incremented / decremented by a margin,
and now again the pixel indices are searched, now around the xFit values with the help of Margin. see below for the code snippet where this is calculated.

```python

	left_lane_inds_incr = ((nonzerox > ((left_fit_Prev[0]*(nonzeroy**2) + left_fit_Prev[1]*nonzeroy + 
                    left_fit_Prev[2]) - margin)) & (nonzerox < ((left_fit_Prev[0]*(nonzeroy**2) + 
                    left_fit_Prev[1]*nonzeroy + left_fit_Prev[2]) + margin))).nonzero()[0]
    right_lane_inds_incr = ((nonzerox > ((right_fit_Prev[0]*(nonzeroy**2) + right_fit_Prev[1]*nonzeroy + 
                    right_fit_Prev[2]) - margin)) & (nonzerox < ((right_fit_Prev[0]*(nonzeroy**2) + 
                    right_fit_Prev[1]*nonzeroy + right_fit_Prev[2]) + margin))).nonzero()[0]
	
```
From the indices values, we get the pixel x and y values by passing the indices again to nonzerox and nonzeroy values.. 
Now the above calculated pixel values are fed to the np.polyfit() function to obtain the new Left and right fit that corresponds to the lanes.

#### 5. Remapping of the lane lines on to the undistorted image

This is done by the function: image_reMap() which takes in the binary warped image and the detected left_fit and right_fits, and the inverse Transform matrix calculated from the Caliberate_camera() function.
First the Left and the right Lanes are cast on to the Binary warped image (please see the image below). This is done by verticaly stacking the (Left_fit - Margin) and ploty, (similarly done for (Left_fit + Margin) and ploty). 
Now both the vertically stacked arrays are now horizontally stacked to form Left Line window array. This is also extended to form a Right_lane window array. Now both these Left and the Right Lanes are cast onto the Binary img using cv2.fillpoly().

![alt text][image17]

next a rectangular region is being mapped out onto the Binary image using a similar method as mentioned above. see the mentioned picture

![alt text][image18]

now the resultant image is inverse transformed using cv2.warpPerspective() and the output is that you get the lane lines and the polygon mapped into the undistorted image. see the figure below

![alt text][image19]

#### 6. Radius of Curvature and Deviation from the Lane centre calculation:

This function takes in the output of generate_data_real() which are left and right fits (corrected ones), and then calculates the Left and the right radius of curvature from the formula:

Rcurve ​= (1+(2Ay+B)2)3/2​
		  -------------
		      |2A|
			  
A and B are polynomial coefficients that are derived from the Left and the Right Fits. we also need to correct the Left and the Right Fits because the polynomial would be calculated in the pixel space and the same needs to be mapped to the real world space.
This is done by multiplying a factor to the xFit datas and ploty datas. as below. The factor represents the transformation from pixel to the real world space.

```python

	ploty = np.linspace(0, image[:,:,0].shape[0]-1, image[:,:,0].shape[0]) * ym_per_pix

    leftx = np.transpose(np.array(leftx[::-1])) * xm_per_pix
    rightx = np.transpose(np.array(rightx[::-1])) * xm_per_pix

    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx, 2)
	
```

Then the generated Left_Fit and right_fit (corrected ones) from the above step is used in calculating the radius of curvature of the left and the Right Lanes by using the Formula mentioned above.

```python

	left_curverad = np.sqrt((np.square((2*left_fit_cr[0]*y_eval) + left_fit_cr[1]) + 1)**3)/(2*left_fit_cr[0])
    right_curverad = np.sqrt((np.square((2*right_fit_cr[0]*y_eval) + right_fit_cr[1]) + 1)**3)/(2*right_fit_cr[0])
	
```

We take the average of the left and the Right Radius to get the overall radius of Curvature.

For calculating the Deviation of the Vehicle from the lane centre , the xfit of the left and the right lanes and also the image centre point is taken to consideration and the below calculation is used to get the value

```python

	xFitLeft_Bottom = xFitLeft[len(xFitLeft) - 1]
    xFitRight_Bottom = xFitRight[len(xFitRight) - 1]
    centre = xFitRight_Bottom - xFitLeft_Bottom
    image_centre = image[:,:,0].shape[1]/2
    Offset = (image_centre - centre) * xm_per_pix

```

![alt text][image20]

---

### Pipeline (video)

Here's a link to my video result
[video1]
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The Pipeline might not work well when:
1. there are some other road lines apart from the lane lines like (if there is a new patch of road being constructed as shown in the challenge video.
2. My pipeline assumes that the input images are of constant size: [720,1280] but now when the input image is of lower size then the mapping for Perspective transform goes wrong and the lanes are not detected.
3. If the lanes curve too much then in the warped image (birdseye view) , the lines touch the left / right side of the plot and hence a 2nd order polynomial might not exactly fit the lanes.. 

Some possible improvements :
1. Maybe instead of Depending on Sobel gradients, if we depend only on color spaces it might work well on the Challenge video. But then not all the lanes are of Yellow and White combination.. 
2. Maybe if we use Deep learning to find the lane lines then the pipeline might work better for all the conditions.
