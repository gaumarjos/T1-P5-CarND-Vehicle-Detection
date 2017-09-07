import numpy as np
import cv2
import glob
import time
import pickle
from collections import deque
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
# from sklearn.cross_validation import train_test_split  # for scikit-learn version <= 0.17
from sklearn.model_selection import train_test_split  # if you are using scikit-learn >= 0.18
from scipy.ndimage.measurements import label

from includes.car_classification import *


class VehicleDetector():
    def __init__(self,
                 classifier,
                 scaler,
                 colorspace='YCrCb',               # Color space used for all features
                 hog_channel='ALL',                # Image channels on which we want to compute HOG
                 hog_orient=9,                     # HOG possible orientations
                 hog_pix_per_cell=8,               # HOG number of px per cell
                 hog_cell_per_block=2,             # HOG number of cells per block
                 spatial_size=(32,32),             # SPATIAL features resize
                 histogram_bins=32,                # HISTOGRAM number of bins
                 show_intermediate_results=False):
        
        self.classifier = classifier
        self.scaler = scaler
        self.colorspace = colorspace
        self.hog_channel = hog_channel
        self.hog_orient = hog_orient
        self.hog_pix_per_cell = hog_pix_per_cell
        self.hog_cell_per_block = hog_cell_per_block
        self.spatial_size = spatial_size
        self.histogram_bins = histogram_bins
        self.show_intermediate_results = show_intermediate_results
        self.show_intermediate_video = False
        
        self.search_settings = (((400, 400+64+64), 1.0, 2),
                                ((380, 380+96+64), 1.5, 2),
                                ((400, 400+128+64), 2.0, 2),
                                ((440, 440+700), 2.5, 2))
        
        # Best heatmap
        self.heatmap = None
        
        # Heatmaps FIFO length
        self.heatmap_fifo_length = 4
        
        # Heatmaps FIFO
        self.heatmap_fifo = deque(maxlen=self.heatmap_fifo_length)
        
        # Centroid FIFO (list of lists)
        # self.centroid_fifo = deque(maxlen=self.heatmap_fifo_length)
        
        # Frame counter
        self.frame_count = 0
        self.full_frame_processing_every = 50
        
        # The left boundary of the window search
        self.xstart = 63
        
        # Mask dilation kernel size
        self.mask_dilation_kernel = np.ones((50, 50))
        
        # Gaussian blur kernel size
        # self.blur_kernel = 3
        
        # Threshold for heatmap
        self.threshold = 2
        
        
    def search_in_image(self, img):
        
        # Create an empty list to receive positive detection windows
        on_windows = []
        
        draw_img = np.copy(img)
        img = img.astype(np.float32) / 255
        
        # Mask on what portion of the image we shall work
        if self.frame_count % self.full_frame_processing_every == 0:
            masking = False
            mask = np.ones_like(img[:, :, 0])
        else:
            masking = True
            # f, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
            mask = np.sum(np.array(self.heatmap_fifo), axis=0)
            mask[(mask > 0)] = 1
            # ax1.imshow(mask)
            # Check if mask is all dark, in this case it doesn't make sense to have a mask
            if np.sum(mask) == 0:
                mask = np.ones_like(img[:, :, 0])
            else:
                mask = cv2.dilate(mask, self.mask_dilation_kernel, iterations=3)
            # ax2.imshow(mask)
        
        # Add info on image
        if 0:
            font = cv2.FONT_HERSHEY_DUPLEX
            if masking:
                text = 'Mask'
            else:
                text = 'Full'
            cv2.putText(draw_img, text, (40,70), font, 1.5, (255,255,255), 2, cv2.LINE_AA)
        
        # Increment frame counter
        self.frame_count += 1
        
        # Run the search for each setting combination
        for ((ystart, ystop), scale, cells_per_step) in self.search_settings:
            
            # Determine the boundaries of the area we want to search
            nonzero = mask.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            if len(nonzeroy) != 0:
                ystart = max(np.min(nonzeroy), ystart)
                ystop = min(np.max(nonzeroy), ystop)
            if len(nonzeroy) != 0:
                xstart = max(np.min(nonzerox), self.xstart)
                xstop = np.max(nonzerox)
            else:
                continue
                
            if xstop <= xstart or ystop <= ystart:
                continue
                
            # Isolate the portion of image we want to work on
            img_tosearch = img[ystart:ystop, xstart:xstop, :]
            
            # Change colorspace as specified
            ctrans_tosearch = change_colorspace(img_tosearch, colorspace=self.colorspace)
            
            # Scale image as specified
            if scale != 1:
                imshape = ctrans_tosearch.shape
                ys = np.int(imshape[1] / scale)
                xs = np.int(imshape[0] / scale)
                if (ys < 1 or xs < 1):
                    continue
                ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

            if ctrans_tosearch.shape[0] < 64 or ctrans_tosearch.shape[1] < 64:
                continue
                
            # Separate channels
            if self.hog_channel == 'ALL':
                ch1 = ctrans_tosearch[:,:,0]
                ch2 = ctrans_tosearch[:,:,1]
                ch3 = ctrans_tosearch[:,:,2]
            else:
                ch1 = ctrans_tosearch[:,:,self.hog_channel]

            # Define blocks and steps as above
            nxblocks = (ch1.shape[1] // self.hog_pix_per_cell) - 1
            nyblocks = (ch1.shape[0] // self.hog_pix_per_cell) - 1
            nfeat_per_block = self.hog_orient * self.hog_cell_per_block ** 2
            
            # 64 was the original sampling rate, with 8 cells and 8 pix per cell
            window_size = 64
            nblocks_per_window = (window_size // self.hog_pix_per_cell) -1
            
            # If masking is enabled, we can try using a smaller overlap (100 - 12.5%)
            if masking and scale < 1.5:
                cells_per_step = 1  # Instead of overlap, define how many cells to step
            nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
            nysteps = (nyblocks - nblocks_per_window) // cells_per_step

            # Compute HOG features over the whole image
            if self.hog_channel == 'ALL':
                hog1 = get_hog_features(ch1, self.hog_orient, self.hog_pix_per_cell, self.hog_cell_per_block, feature_vec=False)
                hog2 = get_hog_features(ch2, self.hog_orient, self.hog_pix_per_cell, self.hog_cell_per_block, feature_vec=False)
                hog3 = get_hog_features(ch3, self.hog_orient, self.hog_pix_per_cell, self.hog_cell_per_block, feature_vec=False)
            else:
                hog1 = get_hog_features(ch1, self.hog_orient, self.hog_pix_per_cell, self.hog_cell_per_block, feature_vec=False)
            
            # Iterate over all windows in the list
            for xb in range(nxsteps + 1):
                for yb in range(nysteps + 1):

                    ypos = yb * cells_per_step
                    xpos = xb * cells_per_step

                    # Extract HOG for this patch
                    if self.hog_channel == 'ALL':
                        hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                        hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                        hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                        hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                    else:
                        hog_features = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()

                    xleft = xpos * self.hog_pix_per_cell
                    ytop = ypos * self.hog_pix_per_cell
                    
                    # Extract the image patch
                    subimg = ctrans_tosearch[ytop:ytop + window_size, xleft:xleft + window_size]

                    # Spatial features
                    spatial_features = bin_spatial(subimg, size=self.spatial_size)

                    # Histogram features
                    hist_features = color_hist(subimg, nbins=self.histogram_bins)

                    # Stack all features
                    features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)

                    # Scale extracted features to be fed to classifier
                    scaled_features = self.scaler.transform(features)

                    # Predict using your classifier
                    prediction = self.classifier.predict(scaled_features)

                    # If positive (prediction == 1) then save the window
                    show_all_windows = 0
                    if prediction == 1 or show_all_windows:
                        xbox_left = xstart + np.int(xleft * scale)
                        xbox_right = xbox_left + np.int(window_size * scale)
                        ybox_top = ystart + np.int(ytop * scale)
                        ybox_bottom = ybox_top + np.int(window_size * scale)
                        on_windows.append(((xbox_left, ybox_top),(xbox_right, ybox_bottom)))
        
        
        # Compute heatmap and update internal status
        self.update_heatmap(draw_img, on_windows)
        
        # Find final boxes from heatmap using label function
        labels = label(self.heatmap)
        bboxes = self.generate_bboxes(labels)
        draw_img = self.draw_boxes(draw_img, bboxes)
        
        if self.show_intermediate_results:
            f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,10))
            
            on_windows_img = self.draw_boxes(draw_img, on_windows)
            ax1.imshow(on_windows_img)
            ax1.set_title('Positive detections')
            
            ax2.imshow(self.heatmap, cmap='hot')
            ax2.set_title('Heatmap (based on the last {} frames)'.format(self.heatmap_fifo_length))
            
            ax3.imshow(draw_img)
            ax3.set_title('Final boxes')
            
        return draw_img


    def update_heatmap(self, draw_img, bbox_list):
        # Create empty heatmap
        this_frame_heatmap = np.zeros_like(draw_img[:, :, 0]).astype(np.float)
        
        # Add all boxes to the heatmat, we now have an heatmap for the whole image
        for box in bbox_list:
            this_frame_heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        
        # Blur the heatmap to have a more continuous behaviour
        """
        this_frame_heatmap = cv2.GaussianBlur(this_frame_heatmap,
                                              (self.blur_kernel, self.blur_kernel), 0)
        """
        
        # We keep that in memory
        self.heatmap_fifo.append(this_frame_heatmap)
        
        # We update the current heatmap with all the elements from the heatmap history
        self.heatmap = np.sum(np.array(self.heatmap_fifo), axis=0)# / self.heatmap_fifo_length
        
        # And threshold it
        self.heatmap[self.heatmap < self.threshold] = 0
        
        # self.heatmap = np.clip(heatmap, 0, 255)
        
        
    # Generate bounding boxes from labels
    def generate_bboxes(self,labels):
        bboxes = []
        # centroids = []
        for labelnr in range(1, labels[1] + 1):
            nonzero = (labels[0] == labelnr).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            bboxes.append(bbox)
            """
            centroid = ((np.min(nonzerox) + np.max(nonzerox)) / 2,
                        (np.min(nonzeroy) + np.max(nonzeroy)) / 2)
            centroids.append(centroid)
            """
        return bboxes
    
    
    @staticmethod
    # Draw bounding boxes
    def draw_boxes(img, bboxes, color=(0,0,255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy
