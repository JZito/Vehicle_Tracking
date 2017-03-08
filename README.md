##Term 1, Project 5
---
The meat of the project exists in helper_functions.py, a collection of functions for classifying and searching the content, and the detection_pipeline.py project, which inputs car footage and outputs video content with bounding boxes drawn around classified vehicles. 

[//]: # (Image References)
[image1a]: ./im_content/hog_color_spaces_plot_8x8.png
[image1b]: ./im_content/hog_color_spaces_plot_16x16.png
[image3a]: ./im_content/hog_orient_1.png
[image3b]: ./im_content/hog_orient_4.png
[image5]: ./im_content/heatmap_plot.png
[image7]: ./im_content/final_output_bboxes.png
[video1]: ./project_video.mp4

---
###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for extracting HOG features is in the get_hog_features() function, defined at line 15 of the helper_functions.py file. It basically wraps around the skimage 'hog()' function, passing our custom parameters. 

Here is an example across different color spaces using HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image1a]


Here is an example across different color spaces using HOG parameters of 'orientations=9', 'pixels_per-cell=(16,16)' and 'cells_per_block=(4,4)':

![alt text][image1b]

Here are a few different amounts of HOG orientations, first 1 and then 4:
![alt text][image3a]
![alt text][image3b]
Eye-opening seeing the directions of the orientations.
The output features are scaled using the 'StandardScaler()' (detection_pipeline.py, line 77). 

There was a lot of playing with results but ultimately I went with an orientation of 9, 8 pixels per cell and 2 cells per block. [Right on the Wikipedia page for HOG](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients#Orientation_binning), it explains that the researchers responsible for popularizing HOG found 9 orientations to be the most effective number in image recognition tasks.


Functions for deriving the color spatial bins and histogram bins  ('bin_spatial()' and 'color_hist()') are in the helper_functions.py file, largely the same as provided by Udacity. I trained all the features (color spatial bins, histogram bins and HOGs) with a linear SVC (line 96 of detection_pipeline.py).

###Sliding Window Search

Just thinking about the relative sizes of cars within the search area we defined, it's unlikely that a tiny car on the horizon is significant, and it seems likely to increase false positives, so we don't want to search scales we don't need. A car taking up an extremely large portion of the frame is a possibility in the case of bad driving or an accident, but is not going to occur here in our test video. Just watching the video, a significant car at any given point could be occupying roughly half to a quarter of the vertical space of our defined search area. Thus, a total search area with a height of 256 pixels, search heights of 64, 96 and 128 would give us a range of windows to search within that frame without wasting space. 
There was also trial and error involved, especially with overlap, seeing which combinations produced the best results. 

The function 'slide_multiple_windows()' (line 242 of helper_functions.py) calls 'slide_window()' (line 152) from Udacity for each window size defined. 

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector. 

---

### Video Implementation

Here's a [link to my video result](./output_tracked_final.mp4)
I implemented code to filter false positives and combine bounding boxes. The functions for this behavior are all defined in the helper_functions.py file. 

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded, ultimately, at a value of 2 ('apply_threshold()', line 260) to filter noisy false positives.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I constructed bounding boxes to cover the area of each blob detected ('draw_labeled_bboxes()', line 263).  

Here's an example result:

![alt text][image5]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This pipeline is likely fragile, as it does not have a good sense of pedestrians, other vehicles, different weather conditions, night driving, etc. Training on and testing with a variety of additional conditions could help make this much more prepared to handle the various objects and obstacles in the real world.
As for me, I understood the concepts of this project pretty well. These older techniques are less trippy and conceptually heavy than neural networks, but unlike neural networks, where the model was easy to construct but the details of what was going in within it was a challenge to understand, putting these functions together into a viable pipeline, arranging the various color features, figuring out how to distingush shapes, devising a threshold for the heat map had me scratching my head. Help from the Udacity community was vital for this one. I feel good about the final result, but there's no question it could be more robust. I would also like to see the image recognition results of a neural network on this content, and I will explore it if I can find the time. 

