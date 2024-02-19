"""
Content-Based Image Retrieval

To run the program with default arguments and say beach_1.jpg as the query image, type the following in the terminal
$ python main.py -q beach_1

The other available arguments are:
-f       : to specify the type of feature - only color histogram (color) or only lbp histogram (lbp) or both (both)
-color   : to specify the method for color histogram - grayscale with 8 bins (gray_8), grayscale with 256 bins (gray_256) and RGB histogram (rgb)
-lbp     : to specify the method for LBP histogram - using whole image (whole_image) or by dividing the image into grids (grid_image)
-dist    : to specify the distance measure used to compare the image feature vectors

A complete example to run the program is thus:
$ python main.py -q beach_1 -f both -color rgb -lbp whole_image -dist euclidean

The starter code is compatible with both python 2 and 3

"""

### Load libraries

import numpy as np
import imageio
import glob, argparse, sys
sys.path.append('images')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def populate_hist(buckets, image):
	# loop through the pixel values and populate the pre-allocated hist_vector
	hist_vector = np.zeros((buckets,))
	# hist_vector = np.zeros(buckets)
	bin_size = 256 / buckets

	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			value = image[i,j]
			index = int (value / bin_size)
			hist_vector[index] += 1

	return hist_vector

def lbp_whole_image(buckets, image):
	# for each pixel p, create an 8 bit number b = 0 if neighbour has value <= p and 1 otherwise
	# initialize variables

	hist_vector = np.zeros((buckets,))
	bin_size = 256 / buckets
	image_h = image.shape[0]
	image_w = image.shape[1]

	# loop through every pixel
	for i in range(image_h):
		for j in range(image_w):
			# ignore boundaries
			if (i == 0 or j ==0 or i == image_h - 1 or j == image_w - 1):
				continue
			pattern = get_lbp(i, j, image)
			index = int(int(pattern, 2) / bin_size)
			hist_vector[index] += 1
	return hist_vector

def lb_grid_image(image):
	# create the big histogram
	num_grids = 12
	hist_vector = np.zeros((32 * num_grids,))
	hist_offset = 0
	# loop through all the 16x16 buckets
	for i in range (4):
		for j in range (3):
			# loop throuh every pixel
			for h in range (16):
				for w in range (16):
					# top left coordinate is image[16 * i,16 * j].
					# the pixel is image[base_i + h, base_j + w]
					curr_i = 16 * i + h
					curr_j = 16 * j + w

					# ignore bounds
					if (curr_i == 0 or curr_j == 0 or curr_i >= image.shape[0] - 1 or curr_j >= image.shape[1] - 1):
						continue
					# get the lbp and update hist_vector
					pattern = get_lbp(curr_i, curr_j, image)
					index = int(int(pattern, 2) / 8) + (hist_offset * 32)
					hist_vector[index] += 1
			hist_offset += 1
	return hist_vector

def get_lbp(i, j, image):
	# given the cooridinate, get one lbp value
	curr = image[i,j]
	pattern = ""

	# top left
	if (image[i-1][j-1] > curr):
		pattern += "1"
	else:
		pattern += "0"

	# top middle
	if (image[i-1][j] > curr):
		pattern +="1"
	else:
		pattern += "0"
	#top right
	if (image[i-1][j+1] > curr):
		pattern +="1"
	else:
		pattern += "0"

	# middle right
	if (image[i][j+1] > curr):
		pattern +="1"
	else:
		pattern += "0"
	# bottom right
	if (image[i+1][j+1] > curr):
		pattern +="1"
	else:
		pattern += "0"
	# bottom middle
	if (image[i+1][j] > curr):
		pattern +="1"
	else:
		pattern += "0"
	# bottom left
	if (image[i+1][j-1] > curr):
		pattern +="1"
	else:
		pattern += "0"
	# middle left
	if (image[i][j-1] > curr):
		pattern +="1"
	else:
		pattern += "0"
	return pattern

### Function to extract color histogram for each image
def color_histogram(image,method):
	# convert RGB image to grayscale
	gray_image = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]

	if method == 'gray_8':
		# hist_vector = np.zeros((8,))
		hist_vector = populate_hist(8, gray_image)

	elif method == 'gray_256':
		# hist_vector = np.zeros((256,))
		hist_vector = populate_hist(256, gray_image)

	elif method == 'rgb':
		num_bins = 256
		hist_vector = np.zeros((num_bins * 3,))
		for i in range(len(image)):
			for j in range(len(image[0])):
				r, g, b = image[i, j, :]
				r_bin = r
				g_bin = g + num_bins
				b_bin = b + num_bins * 2
				hist_vector[r_bin] += 1
				hist_vector[g_bin] += 1
				hist_vector[b_bin] += 1

	else:
		print('Error: incorrect color histogram method')
		return []

	# normalize the histogram (REMOVE THE EPSILON WHEN IMPLEMENTED)
	hist_vector /= sum(hist_vector) # + 1e-10
	return list(hist_vector)

### Function to extract LBP histogram for each image
def lbp_histogram(image,method):
	# convert RGB image to grayscale
	gray_image = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]

	# Now to access pixel (h,w) of gray image, use gray_image[h,w]
	if method == 'whole_image':
		# hist_vector = np.zeros((256,))
		hist_vector =  lbp_whole_image(256, gray_image)

	elif method == 'grid_image':
		# you can remove and rewrite the following 2 lines according to your implementation
		hist_vector = lb_grid_image(gray_image)

	else:
		print('Error: incorrect lbp histogram method')
		return []

	# normalize the histogram (REMOVE THE EPSILON WHEN IMPLEMENTED)
	hist_vector /= sum(hist_vector) #+ 1e-10
	return list(hist_vector)

### Function to compute the feature vector for a given image

def calculate_feature(image, featuretype, color_hist_method, lbp_hist_method):
	# create and return the feature vector as a list
	feature_vector = []

	if featuretype == 'color':
		feature_vector += color_histogram(image, method=color_hist_method)
	elif featuretype == 'lbp':
		feature_vector += lbp_histogram(image, method=lbp_hist_method)
	elif featuretype == 'both':
		feature_vector += color_histogram(image, method=color_hist_method)
		feature_vector += lbp_histogram(image, method=lbp_hist_method)
	else:
		print('Error: incorrect feature type')

	return feature_vector

########## MAIN PROGRAM ##########

if __name__ == "__main__":

	### Provide the name of the query image, for example: beach_1

	ap = argparse.ArgumentParser()
	ap.add_argument("-q", "--query", type=str, required = True, help = "name of the query image")
	ap.add_argument("-f", "--feature", type=str, default = 'both', help = "image feature(s) to be extracted")
	ap.add_argument("-color", "--color_hist_method", type=str, default = 'rgb', help = "method for color histogram")
	ap.add_argument("-lbp", "--lbp_hist_method", type=str, default = 'whole_image', help = "method for lbp histogram")
	ap.add_argument("-dist", "--distance_measure", type=str, default = 'euclidean', help = "distance measure for image comparison")
	args = ap.parse_args()

	### Get all the image names in the database

	image_names = sorted(glob.glob('images/*.jpg'))
	num_images = len(image_names)

	### Create an empty list to hold the feature vectors
	### We shall append each feature vector to this vector
	### Later it will be converted to an array of shape (#images, feature_dimension)

	features = []

	### Loop over each image and extract a feature vector

	for name in image_names:
		print('Extracting feature for ',name.split('/')[-1])
		image = imageio.imread(name)
		feature = calculate_feature(image, args.feature, args.color_hist_method, args.lbp_hist_method)
		features.append(feature)

	### Read the query image and extract its feature vector

	query_image = imageio.imread('images/'+args.query+'.jpg')
	query_feature = calculate_feature(query_image, args.feature, args.color_hist_method, args.lbp_hist_method)

	### Compare the query feature with the database features

	query_feature = np.reshape(np.array(query_feature),(1,len(query_feature)))
	features = np.array(features)
	print('Calculating distances...')
	distances = cdist(query_feature, features, args.distance_measure)

	### Sort the distance values in ascending order

	distances = list(distances[0,:])
	sorted_distances = sorted(distances)
	sorted_imagenames = []

	### Perform retrieval; plot the images and save the result as an image file in the working folder
	fig = plt.figure()

	for i in range(num_images):
		fig.add_subplot(5,8,i+1)
		image_name = image_names[distances.index(sorted_distances[i])]
		sorted_imagenames.append(image_name.split('/')[-1].rstrip('.jpg'))
		plt.imshow(imageio.imread(image_name))
		plt.axis('off')
		plt.title(str(i+1))

	figure_save_name = 'Q_'+args.query+'_F_'+args.feature+'_C_'+args.color_hist_method+'_L_'+args.lbp_hist_method+'_D_'+args.distance_measure+'.png'
	plt.savefig(figure_save_name, bbox_inches='tight')
	plt.close(fig)

	### Calculate and print precision value (in percentage)

	precision = 0
	query_class = args.query.split('_')[0]
	for i in range(5):
		retrieved_class = sorted_imagenames[i].split('_')[0]
		if retrieved_class == 'images\\' + query_class:
			precision += 1
	print('Precision: ',int((precision/5)*100),'%')
