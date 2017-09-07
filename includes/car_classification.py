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


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img,
                                  orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img,
                       orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

    
# Define a function to compute binned color features  
def bin_spatial(img, size):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))


# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Change image color space
def change_colorspace(img, colorspace = 'RGB'):
    if colorspace != 'RGB':
        if colorspace == 'HSV':
            conv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif colorspace == 'LUV':
            conv_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif colorspace == 'HLS':
            conv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif colorspace == 'YUV':
            conv_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif colorspace == 'YCrCb':
            conv_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        conv_image = np.copy(img)
    
    return conv_image


# Extract features from a single image
def extract_features_from_image(img,
                                colorspace='YCrCb',
                                spatial_size=(32, 32),
                                hist_bins=32,
                                orient=9, 
                                pix_per_cell=8,
                                cell_per_block=2,
                                hog_channel='ALL',
                                show_intermediate=False):
    # Define an empty list to receive features
    img_features = []
    
    # Apply color conversion if other than 'RGB'
    feature_image = change_colorspace(img, colorspace)
        
    # Compute spatial features
    spatial_features = bin_spatial(feature_image, size=spatial_size)
    
    # Append features to list
    img_features.append(spatial_features)
    
    # Compute histogram features
    hist_features = color_hist(feature_image, nbins=hist_bins)
    
    # Append features to list
    img_features.append(hist_features)
    
    # Compute HOG features
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                orient, 
                                pix_per_cell, 
                                cell_per_block, 
                                vis=False, 
                                feature_vec=True))
        hog_features = np.array(hog_features)
    else:
        hog_features = get_hog_features(feature_image[:,:,hog_channel], 
                                        orient, 
                                        pix_per_cell, 
                                        cell_per_block, 
                                        vis=False, 
                                        feature_vec=True)
                    
    # Compute HOG features for visualization purpose
    if show_intermediate:
        hot_images = []
        for channel in range(feature_image.shape[2]):
            features, hog_image = (get_hog_features(feature_image[:,:,channel], 
                                   orient,
                                   pix_per_cell,
                                   cell_per_block, 
                                   vis=True,
                                   feature_vec=True))
            hot_images.append(hog_image)
        
    # Append features to list
    img_features.append(hog_features)
    
    # Show intermediate results for debug
    if show_intermediate:
        f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(1,7, figsize=(20,10))
        ax1.imshow(img)
        ax1.set_title('Original')
        
        ax2.imshow(feature_image[:,:,0], cmap='gray')
        ax2.set_title('Y')
        ax3.imshow(feature_image[:,:,1], cmap='gray')
        ax3.set_title('Cr')
        ax4.imshow(feature_image[:,:,2], cmap='gray')
        ax4.set_title('Cb')
        
        ax5.imshow(hot_images[0], cmap='gray')
        ax5.set_title('Y')
        ax6.imshow(hot_images[1], cmap='gray')
        ax6.set_title('Cr')
        ax7.imshow(hot_images[2], cmap='gray')
        ax7.set_title('Cb')
        
    
    # Return concatenated array of features
    return np.concatenate(img_features)


# Extract features from a list of images
def extract_features_from_image_list(img_file_list,
                                     colorspace='YCrCb',
                                     spatial_size=(32, 32),
                                     hist_bins=32,
                                     orient=9,
                                     pix_per_cell=8,
                                     cell_per_block=2,
                                     hog_channel='ALL',
                                     show_intermediate=False):
    # Create a list to append feature vectors to
    features = []
    
    # Iterate through the list of images
    for img_file in img_file_list:      
        # Read in each one by one
        img = mpimg.imread(img_file)
        # Extract features from each of them
        img_features = extract_features_from_image(img,
                                                   colorspace=colorspace,
                                                   spatial_size=spatial_size,
                                                   hist_bins=hist_bins,
                                                   orient=orient, 
                                                   pix_per_cell=pix_per_cell,
                                                   cell_per_block=cell_per_block,
                                                   hog_channel=hog_channel,
                                                   show_intermediate=show_intermediate)
        
        features.append(img_features)
        
    # Return list of feature vectors
    return features
    

# Train a classifier
def train_classifier(cars, notcars, parameters, output_filename):
    
    # Load images and extract features
    t0 = time.time()
    car_features = extract_features_from_image_list(cars,
                                                    colorspace=parameters["colorspace"], 
                                                    spatial_size=parameters["spatial_size"],
                                                    hist_bins=parameters["histogram_bins"], 
                                                    orient=parameters["hog_orient"],
                                                    pix_per_cell=parameters["hog_pix_per_cell"], 
                                                    cell_per_block=parameters["hog_cell_per_block"], 
                                                    hog_channel=parameters["hog_channel"])
    notcar_features = extract_features_from_image_list(notcars,
                                                       colorspace=parameters["colorspace"], 
                                                       spatial_size=parameters["spatial_size"],
                                                       hist_bins=parameters["histogram_bins"], 
                                                       orient=parameters["hog_orient"],
                                                       pix_per_cell=parameters["hog_pix_per_cell"], 
                                                       cell_per_block=parameters["hog_cell_per_block"], 
                                                       hog_channel=parameters["hog_channel"])
    t_extract = time.time() - t0


    # Create the features vector and scale it
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y,
                                                        test_size=0.2,
                                                        random_state=rand_state)

    # Define and fit model
    t0 = time.time()
    print('Feature vector length:', len(X_train[0]))
    svc = LinearSVC(penalty='l2')
    # svc = SVC(kernel="rbf")

    svc.fit(X_train, y_train)
    t_fit = time.time() - t0

    # Check prediction accuracy
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # Check the prediction time for a single sample
    t0 = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t_prediction = time.time() - t0

    # Save model and all processing settings
    model = {"model": svc,
             "scaler": X_scaler,
             "colorspace": parameters["colorspace"],
             "hog_channel": parameters["hog_channel"],
             "hog_orient": parameters["hog_orient"],
             "hog_pix_per_cell": parameters["hog_pix_per_cell"],
             "hog_cell_per_block": parameters["hog_cell_per_block"],
             "spatial_size": parameters["spatial_size"],
             "histogram_bins": parameters["histogram_bins"]}
    output_filename = 'models/' + output_filename + '.p'
    pickle.dump(model, open(output_filename, "wb"))

    print()
    print('Time to load images and extract features: {:6.4f}s'.format(t_extract))
    print('Time to fit model: {:6.4f}s'.format(t_fit))
    print('Time to compute model accuracy: {:6.4f}s for {} predictions'.format(t_prediction, n_predict))
    
