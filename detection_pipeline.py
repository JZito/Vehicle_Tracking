import os
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
import helper_functions as hf

#LOAD CONTENT
plt.ion()
veh_dir = 'vehicles/'

kinds = os.listdir(veh_dir)
vehs = []
for im_type in kinds:
    vehs.extend(glob.glob(veh_dir+im_type+'/*'))

non_veh_dir = 'non-vehicles/'

kinds = os.listdir(non_veh_dir)
non_vehs = []
for im_type in kinds:
    non_vehs.extend(glob.glob(non_veh_dir+im_type+'/*'))

#MAKE LABELS
with open('vehicles.txt', 'w') as v:
    for fn in vehs:
        v.write(fn+'\n')
with open('non-vehicles.txt', 'w') as n:
    for fn in non_vehs:
        n.write(fn+'\n')

### TRAINING
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 656] # Min and max in y to search in slide_window()

t = time.time()
n_samples = 1000
random_idxs = np.random.randint(0,len(vehs), n_samples)

test_v = vehs#np.array(vehs)[random_idxs]
test_nv = non_vehs#np.array(non_vehs)[random_idxs]


car_features = hf.extract_features(test_v, cspace=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spat_features=spatial_feat, 
                        hist_features=hist_feat, hog_features=hog_feat)
notcar_features = hf.extract_features(test_nv, cspace=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spat_features=spatial_feat, 
                        hist_features=hist_feat, hog_features=hog_feat)
print(time.time()-t, 'Seconds to compute features ....')
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

out_images = []
out_maps = []
out_boxes = []
img_boxes = []
ystart = 400
ystop = 656
scale = 1.5

def find_cars(img, scale):
    count = 0
    #for img_src in example_images:
    draw_img = np.copy(img)

    heatmap = np.zeros_like(img[:,:,0])
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = hf.color_swap(img_tosearch, cspace='YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv.resize(ctrans_tosearch, (np.int(imshape[1]/scale),np.int(imshape[0]/scale)) )

    ch0 = ctrans_tosearch[:,:,0]
    ch1 = ctrans_tosearch[:,:,1]
    ch2 = ctrans_tosearch[:,:,2]

    nxblocks = (ch0.shape[1] // pix_per_cell) - 1
    nyblocks = (ch0.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient*cell_per_block**2
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    hog0 = hf.get_hog_features(ch0, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog1 = hf.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = hf.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            count += 1
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            hog_feat0 = hog0[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat0, hog_feat1, hog_feat2))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            #extract the image patch
            subimg = cv.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            #features time
            spatial_features = hf.bin_spatial(subimg, size=spatial_size)
            hist_features = hf.color_hist(subimg, nbins=hist_bins, plot=False)

            #scale features
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features))).reshape(1,-1)
            test_prediction = svc.predict(test_features)

            #if prediction says true, draw a box and add it to the heatmap
            if test_prediction == 1:
                print ("true")
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv.rectangle(draw_img, (xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart), (0,0,255), 6)
                img_boxes.append(((xbox_left, ytop_draw+ystart ), (xbox_left+win_draw, ytop_draw+win_draw+ystart)))
                heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] += 1
        print (time.time()-t, 'seconds to run, total windows = ', count)
    return draw_img, heatmap

#run our process on one image/one frame of video
count = 0

def process_image(img):
	global count
	count += 1
	image = img.astype(np.float32)/255
	draw_image = np.copy(image)
	windows = hf.slide_multiple_windows(image, x_start_stop=[None, None], y_start_stop=y_start_stop, xy_window=(64,64), overlap=(0.5, 0.5))
	hot_windows = hf.search_windows(image, windows, svc, X_scaler, cspace=color_space,
	                         spatial_size=spatial_size, hist_bins=hist_bins,
	                         orient=orient, pix_per_cell=pix_per_cell,
	                         cell_per_block=cell_per_block,
	                         hog_channel=hog_channel, spatial_feat=spatial_feat,
	                         hist_feat=hist_feat, hog_feat=hog_feat)

	window_img = hf.draw_boxes(draw_image, hot_windows, color=(0, 25, 255), thick=6)
	heat = np.zeros_like(window_img[:,:,0]).astype(np.float)
	heat_threshold = 2
	heatmap = hf.add_heat(heat, hot_windows)
	heatmap = hf.apply_threshold(heatmap, heat_threshold)
	if (count < 4):
		cv.imwrite("heatmap"+str(count)+".png", heatmap)
	labels = label(heatmap)
	applied_image = hf.draw_labeled_bboxes(np.copy(img), labels)    
	
	if (count < 4):
		dest_RGB = cv.cvtColor(applied_image, cv.COLOR_BGR2RGB)
		cv.imwrite("final output-" + str(time.time()) + ".png", dest_RGB)
	return applied_image

#create video
input_video = 'test_video.mp4'  
output_video = 'output_final_pipeline_b_test.mp4'

clip1 = VideoFileClip(input_video)
video_clip = clip1.fl_image(process_image)
video_clip.write_videofile(output_video, audio=False)