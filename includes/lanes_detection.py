import numpy as np
import glob
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from includes.lanes_camera_calibration import *


def gradient_magnitude_threshold(img, sobel_kernel=3, thresh=(0, 255)):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(sobelx**2 + sobely**2)
    
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return binary_output


def gradient_direction_threshold(img, sobel_kernel=3, thresh_deg=(0., 90.)):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    dir_sobel = np.arctan2(sobely, sobelx)
    
    thresh = (thresh_deg[0] / 180. * np.pi, thresh_deg[1] / 180. * np.pi)
    binary_output = np.zeros_like(dir_sobel)
    binary_output[(dir_sobel >= thresh[0]) & (dir_sobel <= thresh[1])] = 1
    
    blur_kernel = sobel_kernel
    blurred = cv2.GaussianBlur(binary_output, (blur_kernel, blur_kernel), 0)
    
    return blurred


def gradient_x_threshold(img, sobel_kernel=3, thresh=(0, 255)):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return binary_output


def color_select(img):
    
    rgb_r = img[:,:,0]
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls_h = hls[:,:,0]
    hls_l = hls[:,:,1]
    hls_s = hls[:,:,2]
    
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab_b = lab[:,:,2]
    
    ret, rgb_r_binary = cv2.threshold(rgb_r, 210, 255, cv2.THRESH_BINARY)
    ret, hls_s_binary = cv2.threshold(hls_s, 175, 255, cv2.THRESH_BINARY)
    ret, hls_l_binary = cv2.threshold(hls_l, 200, 255, cv2.THRESH_BINARY)
    hls_l_adapt_binary = cv2.adaptiveThreshold(hls_l,
                                               255,
                                               cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY,
                                               35,
                                               -40)
    ret, lab_b_binary = cv2.threshold(lab_b, 160, 255, cv2.THRESH_BINARY)
    lab_b_adapt_binary = cv2.adaptiveThreshold(lab_b,
                                               255,
                                               cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY,
                                               21,
                                               -6)

    
    # Combine
    combined_binary = np.zeros_like(hls_l_binary)
    """
    combined_binary[(hls_s_binary == 255) |
                    (hls_l_binary == 255) |
                    (lab_b_binary == 255)] = 1
    """
    combined_binary[(hls_l_binary == 255) |
                    (lab_b_adapt_binary == 255)] = 1
    
    # Debug
    if 0:
        if 0:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            ax1.imshow(lab_b, cmap='gray')
            ax1.set_title('lab_b')
            ax2.imshow(lab_b_binary, cmap='gray')
            
        if 1:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            ax1.imshow(lab_b, cmap='gray')
            ax1.set_title('lab_b_adapt')
            ax2.imshow(lab_b_adapt_binary, cmap='gray')
            
        if 1:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            ax1.imshow(hls_l, cmap='gray')
            ax1.set_title('hls_l')
            ax2.imshow(hls_l_binary, cmap='gray')
            
        if 0:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            ax1.imshow(hls_l, cmap='gray')
            ax1.set_title('hls_l_adapt')
            ax2.imshow(hls_l_adapt_binary, cmap='gray')
            
        if 0:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            ax1.imshow(hls_s, cmap='gray')
            ax1.set_title('hls_s')
            ax2.imshow(hls_s_binary, cmap='gray')
            
        if 1:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            ax1.imshow(img)
            ax1.set_title('original')
            ax2.imshow(combined_binary, cmap='gray')
            ax2.set_title('combined')
            
    return combined_binary


def image_processing_pipeline(img, mtx, dist):
    
    # Make a local copy
    img = np.copy(img)
    
    # Produce the undistorted version of the original image. This goes directly to the output.
    undistorted = undistort_image(img, mtx, dist)
    
    # Process the image to highlight lines
    gradient_binary = gradient_x_threshold(img, sobel_kernel=9, thresh=(20, 100))
    color_binary = color_select(img)
    
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    stacked_binary = np.dstack((np.zeros_like(gradient_binary),
                                gradient_binary,
                                color_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(color_binary)
    """
    combined_binary[(gradient_binary == 1) |
                    (color_binary == 1)] = 1
    """
    combined_binary[(color_binary == 1)] = 1
    
    # Region of interest
    # region = region_of_interest(combined_binary)
    
    # Warping
    processed, M, Minv = warp_image(undistort_image(combined_binary, mtx, dist))
    
    # Debug
    if 0:
        print(stacked_binary.shape)
        print(combined_binary.shape)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.set_title('Stacked')
        ax1.imshow(stacked_binary)
        ax2.set_title('Combined')
        ax2.imshow(combined_binary, cmap='gray')
    
    return undistorted, processed, M, Minv
    
    
def lines_fit_new_frame(img, search_area='half', margin=60, minpix = 30, show=False):
    
    # Histogram of the bottom half of the image (where less noisy line data are)
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    #f, ax = plt.subplots()
    #ax.plot(histogram)
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    quarterpoint = np.int(midpoint/2)
    if search_area == 'half':
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    elif search_area == 'quarter':
        leftx_base = np.argmax(histogram[quarterpoint:midpoint]) + quarterpoint
        rightx_base = np.argmax(histogram[midpoint:(midpoint+quarterpoint)]) + midpoint
    
    # Number of sliding windows (vertical)
    nwindows = 10
    
    # Set height of windows
    window_height = np.int(img.shape[0] / nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Rectangle for visualization
    rectangles = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Append the data for the current rectangle. The actual drawing can be done later on if needed.
        rectangles.append((win_y_low, win_y_high,
                           win_xleft_low, win_xleft_high,
                           win_xright_low, win_xright_high))
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit, right_fit = (None, None)
    if len(leftx) is not 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) is not 0:
        right_fit = np.polyfit(righty, rightx, 2)
        
    left_linetype, right_linetype = detect_linetype(leftx, rightx)
    
    # Show results
    if show:
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((img, img, img))*255
        
        # New figure and axis
        f, ax = plt.subplots(1, 1, figsize=(10,5))
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Draw the windows on the visualization image
        for rectangle in rectangles:
            cv2.rectangle(out_img,
                          (rectangle[2],rectangle[0]),
                          (rectangle[3],rectangle[1]),
                          (0,255,0),2) 
            cv2.rectangle(out_img,
                          (rectangle[4],rectangle[0]),
                          (rectangle[5],rectangle[1]),
                          (0,255,0), 2)
        

        # Left line pixels in red and right line pixels in blue
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        # Plot image
        plt.imshow(out_img)
        
        # Fitted lines
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        
    return left_fit, right_fit, left_lane_inds, right_lane_inds, left_linetype, right_linetype


def lines_fit_based_on_previous_frame(img, prev_left_fit, prev_right_fit, margin=80, show=False):

    # Assume you now have a new warped binary image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_lane_inds = ((nonzerox > (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy + prev_left_fit[2] - margin)) & 
                      (nonzerox < (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy + prev_left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy + prev_right_fit[2] - margin)) & 
                       (nonzerox < (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy + prev_right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each
    left_fit, right_fit = (None, None)
    if len(leftx) != 0:
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)
        
    left_linetype, right_linetype = detect_linetype(leftx, rightx)
        
    # Show results
    if show:
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((img, img, img))*255
        window_img = np.zeros_like(out_img)
        
        # New figure and axis
        f, ax = plt.subplots(1, 1, figsize=(10,5))
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Left line pixels in red and right line pixels in blue
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area (based on the old fit)
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        
    return left_fit, right_fit, left_lane_inds, right_lane_inds, left_linetype, right_linetype


def detect_linetype(leftx, rightx, balance_thresh=1.4, continuous_thresh=8000):
    # Detect which line is continuous and which not
    left_npoints = np.float(leftx.shape[0])
    right_npoints = np.float(rightx.shape[0])
    if right_npoints == 0:
        balance = 100
    else:
        balance = left_npoints / right_npoints
    
    # 0 = dashed
    # 1 = continuous
    if balance > balance_thresh:
        return 1., 0.
    elif balance < 1/balance_thresh:
        return 0., 1.
    else:
        if (left_npoints + right_npoints) / 2 > continuous_thresh:
            return 1., 1.
        else:
            return 0., 0.

def lane_width_validation(left_fit, right_fit, h, expected_lane_width=700, expected_lane_width_margin=100):
    lane_width = None
    if left_fit is not None and right_fit is not None:
        left_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
        right_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
        lane_width = abs(right_fit_x_int - left_fit_x_int)
        if abs(expected_lane_width - lane_width) > expected_lane_width_margin:
            left_fit = None
            right_fit = None
    
    return left_fit, right_fit


def curvature_radius_and_distance_from_centre(img, left_fit, right_fit,
                                                   left_lane_inds, right_lane_inds):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
    xm_per_pix = 3.7/700 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    
    # Default values
    left_curverad, right_curverad, center_dist = (0, 0, 0)
    
    # It's important where you want to measure the radius.
    # In this case, I chose the bottom of the image
    curvature_h = img.shape[0]
    ploty = np.linspace(0, curvature_h-1, curvature_h)
    y_eval = np.max(ploty)
  
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Default output values
    left_curve_radius = None
    right_curve_radius = None
    centre_dist = None
    
    if len(leftx) != 0 and len(rightx) != 0:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        
        # Calculate the new radii of curvature in meters
        left_curve_radius = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curve_radius = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
    # Lane width and car centre distance from lane centre
    if left_fit is not None and right_fit is not None:
        car_position = img.shape[1]/2  # based on camera
        left_fit_x_int = left_fit[0]*curvature_h**2 + left_fit[1]*curvature_h + left_fit[2]
        right_fit_x_int = right_fit[0]*curvature_h**2 + right_fit[1]*curvature_h + right_fit[2]
        lane_width = abs(right_fit_x_int - left_fit_x_int) * xm_per_pix
        lane_centre_position = (right_fit_x_int + left_fit_x_int) / 2
        centre_dist = (car_position - lane_centre_position) * xm_per_pix
        
    return left_curve_radius, right_curve_radius, lane_width, centre_dist


def draw_on_image(original_img, binary_img,
                  left_fit, right_fit,
                  left_lane_inds, right_lane_inds,
                  Minv,
                  left_curve_radius=None,
                  right_curve_radius=None,
                  lane_width=None,
                  centre_dist=None,
                  left_linetype=None,
                  right_linetype=None,
                  detection_type=None,
                  show_processed=True,
                  show=False):
    
    # Make a copy of the original image
    original_img_copy = np.copy(original_img)
    if left_fit is None or right_fit is None:
        return original_img_copy
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, original_img.shape[0]-1, original_img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=25)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=25)
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (original_img_copy.shape[1], original_img_copy.shape[0])) 
    
    # Combine the result with the original image
    original_image_with_lines = cv2.addWeighted(original_img_copy, 1, newwarp, 0.3, 0)
    
    # Add curvature radius. The continuous line has more weight than the dashed one.
    font = cv2.FONT_HERSHEY_DUPLEX
    if left_curve_radius is not None and right_curve_radius is not None:
        if left_linetype > 0.5 and right_linetype < 0.5:
            weights = [2.0, 1.0]
        elif left_linetype < 0.5 and right_linetype > 0.5:
            weights = [1.0, 2.0]
        else:
            weights = [1.0, 1.0]
        curve_radius = np.average([left_curve_radius, right_curve_radius], weights=weights)
        
        if curve_radius < 5000:
            text = 'Radius: {:4.0f}m'.format(curve_radius)
        else:
            text = 'Radius: >5000m'
        cv2.putText(original_image_with_lines, text, (40,70), font, 1.5, (255,255,255), 2, cv2.LINE_AA)
    
    # Add lane width, if provided
    if lane_width is not None:
        text = 'Lane width: {:04.2f}m'.format(lane_width)
        # text = 'Lane width: {:04.2f}m ({:04.0f}px)'.format(lane_width, lane_width / (3.7 / 700))
        cv2.putText(original_image_with_lines, text, (40,70+50*1), font, 1.5, (255,255,255), 2, cv2.LINE_AA)
    
    # Add distance from centre, if provided
    if centre_dist is not None:
        if centre_dist > 0:
            direction = 'right'
        else:
            direction = 'left'
        text = 'Car {:04.2f}m {} of centre'.format(abs(centre_dist), direction)
        cv2.putText(original_image_with_lines, text, (40,70+50*2), font, 1.5, (255,255,255), 2, cv2.LINE_AA)
    
    if 0:
        if left_linetype is not None and right_linetype is not None:
            text = 'L: {}  R: {}'.format(left_linetype, right_linetype)
            cv2.putText(original_image_with_lines, text, (40,70+50*3), font, 1.5, (255,255,255), 2, cv2.LINE_AA)
        
    # Show processed image in a box on the upper right corner
    if show_processed:
        
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_img, binary_img, binary_img))*255
        window_img = np.zeros_like(out_img)
        
        # Left line pixels in red and right line pixels in blue
        nonzero = binary_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        # Draw fit line
        margin = 5
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (255, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (255, 255, 0))
        little_img = cv2.addWeighted(out_img, 1, window_img, 1., 0)
        
        little_img = cv2.resize(little_img, None,
                                      fx=0.25, fy=0.25,
                                      interpolation = cv2.INTER_CUBIC)
        original_image_with_lines[40:40 + little_img.shape[0], \
                                  original_image_with_lines.shape[1] - little_img.shape[1] - 40 : \
                                  original_image_with_lines.shape[1] - 40] = little_img
        
        # Add letter to show if it's a new detection or using previous frames
        if detection_type is not None:
            if detection_type:
                text = 'N'
            else:
                text = 'P'
            cv2.putText(original_image_with_lines, text,
                        (original_image_with_lines.shape[1] - 40 - 50, 40 + little_img.shape[0] + 50),
                        font, 1.5, (255,255,255), 2, cv2.LINE_AA)
        
    if show:
        f, ax = plt.subplots()
        ax.imshow(original_image_with_lines)
        
    return original_image_with_lines


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
        self.fits_history = []#[np.array([False])]  
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
        #number of detected pixels
        # self.px_count = None
        
        # line type (0=dashed, 1=continuous)
        self.best_linetype = 0.
        self.linetypes_history = []
           
        # Maximum memory depth
        self.N = 10
    
    def add_fit(self, fit, inds, linetype):
        
        # The last iteration produced a fit
        if fit is not None:
            
            # We already have a best fit in memory
            if self.best_fit is not None:
                self.diffs = abs(fit-self.best_fit)
                
            # The new fit is way too different from the best fit we have in memory,
            # we'll do as if we had no fit. However, an exception 
            if (self.diffs[0] > 0.01 or self.diffs[1] > 1.0 or self.diffs[2] > 1000.0) and \
               len(self.fits_history) > 0:
                # We haven't detected a fit
                self.detected = False
                #print('Discarded as too different: {}'.format(self.diffs))
                
                # We have a list of old fits
                if len(self.fits_history) > 0:
                    # We do with what we have
                    self.best_fit = np.average(self.fits_history, axis=0)
            
            # The new fit is not too different from the best fit we have in memory
            else:
                # We detected a fit
                self.detected = True
                # Append the fit to the list of fits
                self.fits_history.append(fit)
                if len(self.fits_history) > self.N:
                    # Remove last element
                    self.fits_history = self.fits_history[len(self.fits_history)-self.N:]
                # Compute the best fit using the new one
                self.best_fit = np.average(self.fits_history, axis=0)
            
            ## Line type
            # Append the line type to the list of line types
            self.linetypes_history.append(linetype)
            if len(self.linetypes_history) > self.N:
                # Remove last element
                self.linetypes_history = self.linetypes_history[len(self.linetypes_history)-self.N:]
            # Compute the best fit using the new one
            self.best_linetype = np.round(np.average(self.linetypes_history, axis=0))
        
        # The last iteration didn't produce a fit, we need to live with what we have
        else:
            # We haven't detected a fit
            self.detected = False
            
            # We don't want the list to become too old, otherwise the line can get stuck in some position
            # and never recover from there (e.g. challenge video)
            if len(self.fits_history) > 0:
                self.fits_history = self.fits_history[:len(self.fits_history)-1]
            
            # We have a list of old fits
            if len(self.fits_history) > 0:
                # We do with what we have
                self.best_fit = np.average(self.fits_history, axis=0)


# Define a class to manage the detection process
class LaneDetection():

    def __init__(self,
                 mtx,
                 dist):
                 
        # Camera calibration
        self.mtx = mtx
        self.dist = dist
        
        # Lines
        self.left_line = Line()
        self.right_line = Line()
        

    def pipeline(self, img):
        
        image = img.copy()
        
        # Image processing
        undistorted, processed, M, Minv = image_processing_pipeline(image, self.mtx, self.dist)
        
        # Decide whether to work on a new frame or use info from the previous frame
        if not self.left_line.detected or not self.right_line.detected:
            left_fit, \
            right_fit, \
            left_lane_inds, \
            right_lane_inds, \
            left_linetype, \
            right_linetype = lines_fit_new_frame(processed,
                                                 show=False)
            new_detection = True
        else:
            left_fit, \
            right_fit, \
            left_lane_inds, \
            right_lane_inds, \
            left_linetype, \
            right_linetype = lines_fit_based_on_previous_frame(processed,
                                                               prev_left_fit=self.left_line.best_fit,
                                                               prev_right_fit=self.right_line.best_fit,
                                                               show=False)
            new_detection = False
        
        # Now we have a fit, check conditions to decide whether this fit makes sense or not.
        # If it does, add it to the list of good fits in the left_line and right_line objects, else, just
        # say no good fit comes from this frame and let the object decide which of the previous ones makes
        # sense to use.
        
        # Condition: line width must be within certain boundaries to make sense
        left_fit, \
        right_fit = lane_width_validation(left_fit, 
                                          right_fit, 
                                          h=image.shape[0], 
                                          expected_lane_width=700, 
                                          expected_lane_width_margin=150)
        
        # Add the detected line to the Line object
        self.left_line.add_fit(left_fit, left_lane_inds, left_linetype)
        self.right_line.add_fit(right_fit, right_lane_inds, right_linetype)
        
        # Use the best fit, if it's the current one or some previous, is the object's decision
        if self.left_line.best_fit is not None and self.right_line.best_fit is not None:
            
            # Compute curvature and distance from centre
            left_curve_radius, \
            right_curve_radius, \
            lane_width, \
            centre_dist = curvature_radius_and_distance_from_centre(processed,
                                                                    self.left_line.best_fit,
                                                                    self.right_line.best_fit,
                                                                    left_lane_inds,        # possible error?
                                                                    right_lane_inds)       # possible error?
            
            # Produce output
            undistorted_with_lines = draw_on_image(undistorted, processed,
                                                   self.left_line.best_fit,
                                                   self.right_line.best_fit,
                                                   left_lane_inds,
                                                   right_lane_inds,
                                                   Minv,
                                                   left_curve_radius=left_curve_radius,
                                                   right_curve_radius=right_curve_radius,
                                                   lane_width=lane_width,
                                                   centre_dist=centre_dist,
                                                   left_linetype=self.left_line.best_linetype,
                                                   right_linetype=self.right_line.best_linetype,
                                                   detection_type=new_detection,
                                                   show_processed=True,
                                                   show=False)
            
            return undistorted_with_lines
        
        else:
            return undistorted

