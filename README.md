## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car]: ./output_images/car.png
[notcar]: ./output_images/notcar.png
[hog]: ./output_images/hog.png
[windows]: ./output_images/windows.png
[detection]: ./output_images/detection.png
[searcharea]: ./output_images/searcharea.png
[frame1]: ./output_images/frame1.png
[frame2]: ./output_images/frame2.png
[frame3]: ./output_images/frame3.png

## List of files

* Writeup: `README.md`
* Notebook for this project: [vehicles.ipynb](./vehicles.ipynb)
* Notebook merging lane and car detection: [vehicles_and_lane.ipynb](./vehicles_and_lane.ipynb) 
* Python code: see folder `includes`
* Video files: see folder `videos`
* Output images: contained in folder `output_images`, are the ones presented in this writeup.
* Car classification: see `car_classification_datasets` for the dataset and `models` for the trained model(s).

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This document.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in [includes/car_classification.py](./includes/car_classification.py).

The specific code to compute HOG, spatial and histogram features is contained in functions `get_hog_features`, `bin_spatial` and `color_hist` respectively. `change_colorspace` is used to change colorspace and experiment which colorspace produces the best classification and detection results. All these are the called by `extract_features_from_image` to extract the feature set on one image. This operation is then repeated in function `extract_features_from_image_list` on all images in the `cars` and `notcars` datasets.

The whole dataset available to train and test the classifier was composed of 8792 car and 8968 non-car images. 20% of these were set apart for testing. Here's an example:

![alt text][car] ![alt text][notcar]

I have tested HOG features computed with different colorspaces, channels on which HOG was applied, number of orientation bins, pixel per cell and cell per block. Here is an example using the `YCrCb` color space , computing HOG on all channels with `hog_orient=9`, `hog_pix_per_cell=8` and `hog_cell_per_block=2`:

![alt text][hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of `colorspace`, `hog_channel`, `hog_orient`, `hog_pix_per_cell` and `hog_cell_per_block`, both comparing the classifier prediction accuracy computed on the test dataset and by observing their performance on portions of the video. The latter rather than the former proved more useful to evaluate of "good" the classifier was. In fact, while almost all tested combinations gave classification scores >0.97, their effectiveness in actually detecting cars in video frames was quite different.

In particular, I tried:
* `colorspace` YUV, YCrCb
* `hog_channel` 0, 'ALL'
* `hog_orient` 9, 11
* `hog_pix_per_cell`= 8, 16
* `hog_cell_per_block`= 2

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in function `train_classifier` in file [includes/car_classification.py](./includes/car_classification.py).

In this function features from the both `cars` and `notcars` datasets are extracted and scaled, labels (0 or 1) are assigned, train and test datasets are created and a linear SVM classifier is fitted. Both SVM with linear and rbf kernels were tried. Eventually, the linear solution (`LinearSVC`) was chosen for speed reasons. The whole processing is already slow enough. The prediction score on the test dataset is then computed and the obtained model, scaler and feature parameters are saved in a pickle file to be used later by the car detection algorthm.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in class `VehicleDetector` in file [includes/car_detection.py](./includes/car_detection.py). The specific method is `search_in_image`.

The sliding windows search I implemented uses 4 different window sizes in the 4 different areas of the image, with scales of 1.0, 1.5, 2.0 and 2.5 (with respect to the default size of 64x64px). Smaller scales are used to detect cars further away on the horizon, increasing them as closer the cars are to the observer. Windows are 75% overlapping, this value was chosen as a compromise between a more accurate localization (higher overlapping) and an acceptable processing time (less windows, so less overlapping). A graphical description of the windows used is shown as follows.

![alt text][windows]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In order to obtain a more accurate localization of the cars while speeding up the processing, this kind of search covering the whole road area is actually performed only once every N frames (50 looked like a good compromise after a few tests on the video stream). In all other cases, the search area is limited to the area around which the last detections were. When a limited area is used, the number of search windows is reduced, hence speeding up the search. As I have observed that an overlap larger than 75%, 87.5%, is beneficial to detect cars that are further away, I temporarily increase it when searching in a limited area with the smallest window size (1.0x 64x64px). This produced smoother and less flickery localizations of distant cars. Here is an example of this search area:

![alt text][searcharea]

Another step in the direction of a faster algorithm included, as described in the lessons, using an image-wide HOG feature extraction instead of rerunning the algorithm on each window.

Here are some detection examples:

![alt text][detection]

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to the video result for this project](./videos/project_video_output.mp4)

Here's a [link to the video result for lane and vehicle detection](./videos/project_video_vehicles_and_lane.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is contained in class `VehicleDetector` in file [includes/car_detection.py](./includes/car_detection.py). The specific methods are `search_in_image`, `update_heatmap` and `generate_bboxes`.

Once a frame has been searched for cars (either processing the entire frame or just a limited area), each new detection (so-called "on" windows, where a car has been detected) is summed up in a frame heatmap and this information is appended to a FIFO list of the last N frames (a depth of 4 proved a good compromise between reaction time and smoothing effect). The best heatmap for this frame is then computed as the threesholded sum of all the elements in this FIFO list. In between, I also considered using gaussian blur to blur the heatmap to further smoothen the result, but it hasn't been adding much so I excluded it from the final version.

Although quite basic, I think this simple approach is quite effective as it acts as a spatial running average. I also considered other approaches involving detecting centroids of the hot areas and then tracking them but proved to be complicated to apply and quite tricky when one vehicle overtakes and covers another.

The so-obtained heatmaps were labelled using `scipy.ndimage.measurements.label` and then bounding boxes were generated and drawn on the original camera image.

To summarize the whole process, I searched on 4 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector. I did that using a dynamic limited search area (reset every N frames). After that, I averaged (both to smoothen and to remove false positives) the final results by averaging over the last 4 frames.

Here is an example of 3 consecutive frames:

![alt text][frame1]
![alt text][frame2]
![alt text][frame3]

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

A few problems affecting this implementation and ideas for development are:
* The whole processing is rather slow and I can't imagine it working in real time. Writing it in C++ would definitely help but I think it's also the algorithm structure that needs to be improved.
* A better classifier can be used, but, for example, moving from a linear SVM to one with a RBF kernel would slow down the processing even further. I believe a key improvement in this direction would be using a larger dataset to train the classifier.
* It's difficult to correctly detect and track vehicles that are further away and are whitish/greyish as they can be easily mistaken for asphalt or road infrastructure. I addressed that using larger overlap (i.e. more windows), as in this way I expected heatmap peaks where the actual cars were, but I believe there is room for improvement.
