import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import numpy as np
import json
import pdb
from multiprocessing import Process
import argparse
import cv2
import time
import math
import sys

#--=---=---=---=---=---=---=--

#Testing


parser = argparse.ArgumentParser()
parser.add_argument('--num_pics_per_class',default=1,type=int,help='num images to generate per class')
parser.add_argument('--image_length',default=150,type=int,help='Length dimension of output images')
parser.add_argument('--image_height',default=150,type=int,help='Height dimension of output images')
parser.add_argument('--continue_bool', default=False, type=bool, help='If specified as True, will attempt to continue previous job from where it left off')		# <-- DO NOT USE THIS.. it doesn't work
parser.add_argument('--datadir', default = 'generated-for-classification2', type=str,help='Dir where to put images and labels')
parser.add_argument('--upright_bool', default = False, type=bool,help='If True, only upright targets will be generated')
parser.add_argument('--target_offset_spread', default = 0.1, type=float, help='How much the target is offset according to a normal distribution')

#--=---=---=---=---=---=---=--

args = parser.parse_args()

NUM_IMAGES_PER_CLASS_TO_GENERATE = args.num_pics_per_class
IMAGE_LENGTH = 150
IMAGE_HEIGHT = 150

image_length = args.image_length
image_height = args.image_height
CONTINUE_FROM_SAVED_STATE = args.continue_bool
UPRIGHT_ONLY = args.upright_bool



PARENT_SAVE_DIR = args.datadir+'/'
PIC_SAVE_DIR = PARENT_SAVE_DIR + "images/"
LABEL_SAVE_DIR = PARENT_SAVE_DIR + "labels/"
ANNOTATIONS_FILEPATH = PARENT_SAVE_DIR + "annotations.json"
STATE_FILEPATH = PARENT_SAVE_DIR + ".state.json"

#--=---=---=---=---=---=---=--


# If the following directories don't exist, create them
if not os.path.exists(PARENT_SAVE_DIR):
	os.mkdir(PARENT_SAVE_DIR)
if not os.path.exists(PIC_SAVE_DIR):
	os.mkdir(PIC_SAVE_DIR)
if not os.path.exists(LABEL_SAVE_DIR):
	os.mkdir(LABEL_SAVE_DIR)


#--=---=---=---=---=---=---=--


'''
HYPERPARAMETERS FOR RANDOMIZED PARAMETERS DEFINED HERE
Every time this script generates an image, it randomizes several parameters, such as rotation angle, size, font, etc.
The hyperparameters constraining the choices/ranges/directory-paths for these parameters are defined here. Feel free to edit the hyperparameters.
Make sure you read the comments before editing any hyperparameters, otherwise you might break something
'''
SHAPE_DIR = "Shapes3/"

#--=---=---=---=---=---=---=--

# We only need to generate uppercase letters and numbers.
ALPHANUMERIC_CHOICES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#--=---=---=---=---=---=---=--


print(len(ALPHANUMERIC_CHOICES))

COLOR_DIR = "Colors/"
FONT_DIR = "Fonts_Old/Fonts/"
FONT_SIZE_MINMAX = [10, None]
ALPHANUMERIC_OFFSET_FROM_CENTER_RANGE = range(-1, 2)
ROTATION_ANGLE_RANGE = range(0, 360)
TARGET_SIZE_RANGE = range(40, 148)
MAX_TARGET_SIZE = np.max(TARGET_SIZE_RANGE)
NOISE_DISTRIBUTION_MEAN_RANGE = range(-10, 11)
NOISE_DISTRIBUTION_STDDEV_RANGE = range(20, 40)
BACKGROUND_DIR = "Backgrounds/"
BACKGROUND_CROP_VERTEX_LEFT_RANGE = None
BACKGROUND_CROP_VERTEX_TOP_RANGE = None
BACKGROUND_ROTATION_ANGLE_CHOICES = [0, 180]
BACKGROUND_BRIGHTNESS_MULTIPLIER_RANGE = np.arange(0.5, 1, 0.02)
TARGET_X_OFFSET_MINMAX = [0, None]
TARGET_Y_OFFSET_MINMAX = [0, None]
TARGET_OFFSET_SPREAD = args.target_offset_spread
SHADOW_SPREAD = 1.075
SHADOW_OFFSET = 1

#--=---=---=---=---=---=---=--

#This code says the number of pixels in the image, and the max number of pixels in the target.
#There is then a conditional statement which makes sure that the pixel count in the target is not greater than the actual image.

num_pixels_in_image = IMAGE_LENGTH * IMAGE_HEIGHT
num_pixels_in_target_max = TARGET_SIZE_RANGE[-1] * TARGET_SIZE_RANGE[-1]
if num_pixels_in_target_max > num_pixels_in_image:
	print("!!!!!!!!!!!!!Error: target will not fit in image, either make target smaller or image larger!!!!!!!!!!!!!!!!!")
	sys.exit(1)

#--=---=---=---=---=---=---=--


def choose_random_image_parameters():
	# Only choose background files that don't start with "." (for example hidden files that Github throws in)
	possible_background_files = [file for file in os.listdir(BACKGROUND_DIR) if not file.startswith(".")]
	background_filename = random.choice(possible_background_files)
	# Can't choose the left/top crop vertices here. Need to know the size of the full background image
	background_crop_vertex_left = None
	background_crop_vertex_top = None
	background_rotation_angle = random.choice(BACKGROUND_ROTATION_ANGLE_CHOICES)
	background_brightness_multiplier = random.choice(BACKGROUND_BRIGHTNESS_MULTIPLIER_RANGE)
	randomized_image_params = {
				"background_filename": background_filename,
				"background_crop_vertex_left": background_crop_vertex_left,
				"background_crop_vertex_top": background_crop_vertex_top,
				"background_rotation_angle": background_rotation_angle,
				"background_brightness_multiplier": background_brightness_multiplier,
				}
	return randomized_image_params


#--=---=---=---=---=---=---=--

#These colors are difficult to distinguish. We are not holding our neural network responsible for these.

illegal_color_combinations = [ #(shape_color, text_color)
	("white", "gray"),
	("gray", "white"),
	("gray", "brown"),
	("brown", "gray")
	]

#--=---=---=---=---=---=---=--

#This function chooses the random target parameters and sends them to generate the background

def choose_random_target_parameters(**kwargs):
	# Only choose shape "directories" that don't start with "." (for example hidden files that Github throws in)
	possible_shape_class_dirs = [folder for folder in os.listdir(SHAPE_DIR) if not folder.startswith(".")]
	shape_class_dir = random.choice(possible_shape_class_dirs) if kwargs.get("shape_class_dir") == None else kwargs.get("shape_class_dir")
	shape_filename = random.choice(os.listdir(SHAPE_DIR + shape_class_dir + "/"))
	shape_color_class_dir = random.choice(os.listdir(COLOR_DIR))
	shape_color_filename = random.choice(os.listdir(COLOR_DIR + shape_color_class_dir + "/"))
	alphanumeric = random.choice(ALPHANUMERIC_CHOICES) if kwargs.get("alphanumeric") == None else kwargs.get("alphanumeric")
	alphanumeric_color_class_dir = random.choice(os.listdir(COLOR_DIR))
	while alphanumeric_color_class_dir == shape_color_class_dir or (shape_color_class_dir, alphanumeric_color_class_dir) in illegal_color_combinations:
		# if (shape_color_class_dir, alphanumeric_color_class_dir) in illegal_color_combinations:
		# 	print((shape_color_class_dir, alphanumeric_color_class_dir))
		alphanumeric_color_class_dir = random.choice(os.listdir(COLOR_DIR))
	alphanumeric_color_filename = random.choice(os.listdir(COLOR_DIR + alphanumeric_color_class_dir + "/"))
	font_filename = random.choice(os.listdir(FONT_DIR))
	# Cannot choose font size here.
	font_size = None
	rotation_angle = random.choice(ROTATION_ANGLE_RANGE)
	alphanumeric_offset_from_center_x = random.choice(ALPHANUMERIC_OFFSET_FROM_CENTER_RANGE)
	alphanumeric_offset_from_center_y = random.choice(ALPHANUMERIC_OFFSET_FROM_CENTER_RANGE)
	target_size = random.choice(TARGET_SIZE_RANGE)
	noise_distribution_mean_1 = random.choice(NOISE_DISTRIBUTION_MEAN_RANGE)
	noise_distribution_stddev_1 = random.choice(NOISE_DISTRIBUTION_STDDEV_RANGE)
	noise_distribution_mean_2 = random.choice(NOISE_DISTRIBUTION_MEAN_RANGE)
	noise_distribution_stddev_2 = random.choice(NOISE_DISTRIBUTION_STDDEV_RANGE)
	noise_distribution_mean_3 = random.choice(NOISE_DISTRIBUTION_MEAN_RANGE)
	noise_distribution_stddev_3 = random.choice(NOISE_DISTRIBUTION_STDDEV_RANGE)
	# Can't choose the x_offset and y_offset here.
	target_x_offset = None
	target_y_offset = None
	randomized_target_params = {"shape_class_dir": shape_class_dir,
						"shape_filename": shape_filename,
						"shape_color_class_dir": shape_color_class_dir,
						"shape_color_filename": shape_color_filename,
						"alphanumeric_color_class_dir": alphanumeric_color_class_dir,
						"alphanumeric_color_filename": alphanumeric_color_filename,
						"font_filename": font_filename,
						"font_size": font_size,
						"alphanumeric": alphanumeric,
						"alphanumeric_offset_from_center_x": alphanumeric_offset_from_center_x,
						"alphanumeric_offset_from_center_y": alphanumeric_offset_from_center_y,
						"rotation_angle": rotation_angle,
						"target_size": target_size,
						"noise_distribution_mean_1": noise_distribution_mean_1,
						"noise_distribution_stddev_1": noise_distribution_stddev_1,
						"noise_distribution_mean_2": noise_distribution_mean_2,
						"noise_distribution_stddev_2": noise_distribution_stddev_2,
						"noise_distribution_mean_3": noise_distribution_mean_3,
						"noise_distribution_stddev_3": noise_distribution_stddev_3,
						"target_x_offset": target_x_offset,
						"target_y_offset": target_y_offset,
						"shadow_direction": "" 
						}
	return randomized_target_params

#--=---=---=---=---=---=---=--


def generate_background(randomized_image_params):
	'''
	Unpack the randomized params
	'''
	background_filename = randomized_image_params["background_filename"]
	background_crop_vertex_left = randomized_image_params["background_crop_vertex_left"]
	background_crop_vertex_top = randomized_image_params["background_crop_vertex_top"]
	background_rotation_angle = randomized_image_params["background_rotation_angle"]
	background_brightness_multiplier = randomized_image_params["background_brightness_multiplier"]
	'''
	Open the background, execute some operations on it
		1) Open the background image
		2) Crop the background to IMAGE_LENGTH and height
		3) Rotate the background
		4) Apply a brightness diminisher to the background
	'''
	background = Image.open(BACKGROUND_DIR+background_filename)
	# Now that we know the size of the full background image, we can choose left/top crop vertices. Only do this if the vertex values are not None (will always be the case, unless if the user is debugging)
	if background_crop_vertex_left == None:
		background_crop_vertex_left = random.choice(range(0, background.width-IMAGE_LENGTH))
		randomized_image_params["background_crop_vertex_left"] = background_crop_vertex_left
	if background_crop_vertex_top == None:
		background_crop_vertex_top = random.choice(range(0, background.height-IMAGE_HEIGHT))
		randomized_image_params["background_crop_vertex_top"] = background_crop_vertex_top
	background = background.crop((background_crop_vertex_left, background_crop_vertex_top, background_crop_vertex_left+IMAGE_LENGTH, background_crop_vertex_top+IMAGE_HEIGHT))
	background = background.rotate(background_rotation_angle)
	background = ImageEnhance.Brightness(background).enhance(background_brightness_multiplier)

	return background


def lighting_gradient(target, target_length, target_height, darkness):
	'''
	Applies a small radial gradient effect
	Radially outwards by adding a transparency mask linearly increasing from 0 to darkness*256
	'''
	centerX = np.random.randint(3 * image_length) - 1*image_length
	centerY = np.random.randint(3 * image_height) - 1*image_height

	innerColor = np.asarray([0,0,0,0]) 					  #Color at the center
	outerColor = np.asarray([0, 0, 0, darkness*256]) #Color at the corners

	#Maximum distance from the center to the edge of the target
	maxX = max(abs(centerX - 0), abs(centerX - target_length))
	maxY = max(abs(centerY - 0), abs(centerY - target_height))
	maxDistance = math.sqrt(maxX ** 2 + maxY ** 2)

	target_raw = np.asarray(target)

	Gradient = np.zeros([target_height, target_length, 4])

	for y in range(target_height):
		for x in range(target_length):

			#Find the distance to the center
			distanceToCenter = math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2)

			#Normalize
			distanceToCenter = float(distanceToCenter) / float(maxDistance)

			#Calculate pixel value
			color = outerColor * distanceToCenter + innerColor * (1 - distanceToCenter)
			pixel = color.astype(int)

			# If the target has no value, should the Gradient
			if (target_raw[y,x,3] == 0):
				pixel = innerColor


			#Place the pixel
			Gradient[y,x,:] = pixel

	# Paste the Gradient onto the Target
	Gradient = Gradient.astype(np.uint8)
	target_raw = target_raw.astype(np.uint8)

	Gradient = Image.fromarray(Gradient)
	target = Image.fromarray(target_raw)
	target.paste(Gradient, (0, 0), mask= Gradient)
	return target




#This function generates the randomized target given the randomized target parameters.
def generate_target(randomized_target_params):
	'''
	Step 1: Unpack the randomized params
	'''
	shape_class_dir = randomized_target_params["shape_class_dir"]
	shape_filename = randomized_target_params["shape_filename"]
	shape_color_class_dir = randomized_target_params["shape_color_class_dir"]
	shape_color_filename = randomized_target_params["shape_color_filename"]
	alphanumeric_color_class_dir = randomized_target_params["alphanumeric_color_class_dir"]
	alphanumeric_color_filename = randomized_target_params["alphanumeric_color_filename"]
	font_filename = randomized_target_params["font_filename"]
	font_size = randomized_target_params["font_size"]
	alphanumeric = randomized_target_params["alphanumeric"]
	alphanumeric_offset_from_center_x = randomized_target_params["alphanumeric_offset_from_center_x"]
	alphanumeric_offset_from_center_y = randomized_target_params["alphanumeric_offset_from_center_y"]
	rotation_angle = randomized_target_params["rotation_angle"]
	target_size = randomized_target_params["target_size"]
	noise_distribution_mean_1 = randomized_target_params["noise_distribution_mean_1"]
	noise_distribution_stddev_1 = randomized_target_params["noise_distribution_stddev_1"]
	noise_distribution_mean_2 = randomized_target_params["noise_distribution_mean_2"]
	noise_distribution_stddev_2 = randomized_target_params["noise_distribution_stddev_2"]
	noise_distribution_mean_3 = randomized_target_params["noise_distribution_mean_3"]
	noise_distribution_stddev_3 = randomized_target_params["noise_distribution_stddev_3"]
	'''
	Step 2: Execute all the pre-alphanumeric-draw operations on the target container
		1) Open the target shape image
		2) Create a blank image (target_container) that everything will be pasted into
		3) Draw a shadow in the shape of the target image
		4) Color the shadow
		5) Paste the target image over top of the shadow
		6) Color the target image
	'''

	# Randomly choose a direction to apply the shadow offset. For negative offsets in x and y, 
	# it is easier to offset the target instead of the shadow.
	direction = np.random.randint(0,4)
	shadow_x_offset = 0
	shadow_y_offset = 0
	target_x_offset = 0
	target_y_offset = 0
	if direction == 0:
		shadow_x_offset = SHADOW_OFFSET
		randomized_target_params["shadow_direction"] = "East"
	elif direction == 1:
		shadow_y_offset = SHADOW_OFFSET
		randomized_target_params["shadow_direction"] = "South"
	elif direction == 2:
		target_x_offset = SHADOW_OFFSET
		randomized_target_params["shadow_direction"] = "West"
	else:
		target_y_offset = SHADOW_OFFSET
		randomized_target_params["shadow_direction"] = "North"

	target_image = Image.open(SHAPE_DIR+shape_class_dir+"/"+shape_filename)

	target_container = Image.new("RGBA", (int(target_image.size[0]*SHADOW_SPREAD)+shadow_x_offset+target_x_offset,int(target_image.size[1]*SHADOW_SPREAD)+shadow_y_offset+target_y_offset), (0,0,0,0))
	shadowImage = Image.open(SHAPE_DIR+shape_class_dir+"/"+shape_filename)
	shadow_dx_adjustment = int((shadowImage.size[0] * (SHADOW_SPREAD - 1)) / 2)
	shadow_dy_adjustment = int((shadowImage.size[1] * (SHADOW_SPREAD - 1)) / 2)
	shadowImage = shadowImage.resize((int(shadowImage.size[0]*SHADOW_SPREAD), int(shadowImage.size[1]*SHADOW_SPREAD)), Image.ANTIALIAS)
	shadowImage.paste((0,0,0,255), (0,0,shadowImage.size[0],shadowImage.size[1]), mask=shadowImage)
	target_container.paste(shadowImage, (shadow_x_offset,shadow_y_offset), mask=shadowImage)
	
	target_container.paste(target_image, (shadow_dx_adjustment+target_x_offset,shadow_dy_adjustment+target_y_offset), mask=target_image)
	with open(COLOR_DIR+shape_color_class_dir+"/"+shape_color_filename, 'r') as shape_color_file:
		shape_color = json.load(shape_color_file)
	shape_color.append(255)
	shape_color = tuple(shape_color)
	target_container.paste(shape_color, (shadow_dx_adjustment+target_x_offset,shadow_dy_adjustment+target_y_offset), mask=target_image)

	# temp = np.asarray(target_container)
	# temp = cv2.GaussianBlur(temp,(5,5),0)
	# target_container = Image.fromarray(temp)

	'''
	Step 3: Draw the alphanumeric on the target container and execute the post-alphanumeric-draw operations:
		1) Calculate the maximum allowable font size and choose a font size
			a) Predict the aspect ratio of the alphanumeric
			b) Find the largest rectangle (with an aspect ratio equal to the aspect ratio of the alphanumeric) that fits inside the shape
			c) Find the largest font size that allows the alphanumeric to fit inside that rectangle
			d) Choose a fontsize
		2) Predict the pixel-size of the alphanumeric
		3) Use the predicted pixel-size to determine where the alphanumeric should be drawn so that it is centered on the target container. By centering the alphanumeric on the target container, it will also be centered on the shape.
		4) Draw the alphanumeric
		5) Rotate the target container
	'''


	# Now that we know which font and alphanumeric we are using, and drew the shape, we can calculate the maximum allowable font size and choose one. Only do this if font_size is None (will usually be the case, unless if user is debugging)
	if font_size == None and FONT_SIZE_MINMAX[1] == None:
		alphanumeric_testsize = np.array(ImageFont.truetype(FONT_DIR+font_filename, 100).getsize(alphanumeric))
		alphanumeric_aspect_ratio = alphanumeric_testsize / min(alphanumeric_testsize) * 2
		alphanumeric_maxsize = None
		target_container_center_x = int(target_container.width/2)
		target_container_center_y = int(target_container.height/2)
		target_container_alpha_channel_raw = np.asarray(target_container.split()[-1])
		i=1
		while True:
			alphanumeric_maxsize_candidate = alphanumeric_aspect_ratio * i
			left = target_container_center_x - int((alphanumeric_maxsize_candidate[0]+0.5) / 2)
			right = target_container_center_x + int((alphanumeric_maxsize_candidate[0]+1.5) / 2)
			top = target_container_center_y - int((alphanumeric_maxsize_candidate[1]+0.5) / 2)
			bottom = target_container_center_y + int((alphanumeric_maxsize_candidate[1]+1.5) / 2)
			if np.any(target_container_alpha_channel_raw[top:bottom+1,left:right+1] != 255):
				alphanumeric_maxsize = (right-left-2, bottom-top-2)
				break
			i += 1
		# Now that we know the dimensions of the largest rectangle that fits inside the shape, we test different fontsizes until we find the largest one that doesn't cause the alphanumeric to be bigger than the rectangle. The algorithm doesn't account for whitespace in the alphanumeric.
		font_size_max = None
		font_size_max_candidate = FONT_SIZE_MINMAX[0] + 1
		while True:
			if ImageFont.truetype(FONT_DIR+font_filename,font_size_max_candidate).getsize(alphanumeric)[0] >= alphanumeric_maxsize[0]:
				font_size_max = font_size_max_candidate
				break
			font_size_max_candidate += 1
		# Choose a font size (since we now know the minumum and maximum allowable font sizes)
		font_size = random.choice(range(max(FONT_SIZE_MINMAX[0],int(font_size_max/4 * 3)), font_size_max))
		randomized_target_params["font_size"] = font_size
	elif font_size == None and FONT_SIZE_MINMAX[1] != None:
		font_size = random.choice(range(max(FONT_SIZE_MINMAX[0],int(FONT_SIZE_MINMAX[1]/4 * 3)), FONT_SIZE_MINMAX[1]))
		randomized_target_params["font_size"] = font_size
	font = ImageFont.truetype(FONT_DIR+font_filename, font_size)
	alphanumeric_width_including_whitespace, alphanumeric_height_including_whitespace = font.getsize(alphanumeric)
	alphanumeric_whitespace_x, alphanumeric_whitespace_y = font.getoffset(alphanumeric)
	alphanumeric_width = alphanumeric_width_including_whitespace - alphanumeric_whitespace_x
	alphanumeric_height = alphanumeric_height_including_whitespace - alphanumeric_whitespace_y
	draw_position_x = (target_container.width / 2) - alphanumeric_whitespace_x - (alphanumeric_width / 2) + alphanumeric_offset_from_center_x
	draw_position_y = (target_container.height / 2) - alphanumeric_whitespace_y - (alphanumeric_height / 2) + alphanumeric_offset_from_center_y

	# Make a blank image to paste the alphanumeric onto
	alphanumericImage = Image.new("RGBA", (target_container.width, target_container.height), (128,128,128,0))


	drawer = ImageDraw.Draw(alphanumericImage)
	with open(COLOR_DIR+alphanumeric_color_class_dir+"/"+alphanumeric_color_filename, 'r') as alphanumeric_color_file:
		alphanumeric_color_rgb = json.load(alphanumeric_color_file)
	drawer.text((draw_position_x, draw_position_y), alphanumeric, fill=tuple(alphanumeric_color_rgb), font=font)

	# Blur the thing
	temp = np.asarray(alphanumericImage)
	# temp = cv2.GaussianBlur(temp,(3,3),0)
	alphanumericImage = Image.fromarray(temp)

	# Paste the alphanumeric onto the shape
	target_container.paste(alphanumericImage, (0, 0), mask= alphanumericImage)

	if (not UPRIGHT_ONLY):
		target_container = target_container.rotate(rotation_angle, expand=True)

	'''
	Step 4: Determine the target's dimensions and position relative to the target container (note: this is pretty stupid but I couldn't think of anything better) and use that information to choose the x and y offsets
		1) Search through the target container for the leftmost, topmost, rightmost, and bottommost pixels that are part of the target in order to determine the x-origin, y-origin, length, and height (this is pretty stupid that I have to do this)
		2) Crop out the target
		3) Resize the target
		4) Choose the x and y offsets (we can do this now because we know the final size of the target)
	'''

	# Leave a few pixels of padding around the target to prevent the edges from being cropped off
	target_padding = 5

	target_container_alpha_channel = target_container.split()[-1]
	target_container_length = target_container.size[0]
	target_container_height = target_container.size[1]
	# Search through the target container to determine the x-origin of the target relative to the target container
	target_x_origin_relative_to_container = -1
	search_complete = False
	for i in range(target_container_length):
		if search_complete:
			break
		for j in range(target_container_height):
			if target_container_alpha_channel.getpixel((i, j)) > 10:
				target_x_origin_relative_to_container = i - target_padding
				search_complete = True
				break
	# Search through the target container to determine the y-origin of the target relative to the target container
	target_y_origin_relative_to_container = -1
	search_complete = False
	for i in range(target_container_height):
		if search_complete:
			break
		for j in range(target_container_length):
			if target_container_alpha_channel.getpixel((j, i)) > 10:
				target_y_origin_relative_to_container = i - target_padding
				search_complete = True
				break
	# Search through the target container to find rightmost target pixel and calculate the length of the target
	target_length_before_resize = -1
	search_complete = False
	for i in range(target_container_length):
		if search_complete:
			break
		for j in range(target_container_height):
			if target_container_alpha_channel.getpixel((target_container_length-1-i, j)) > 10:
				target_length_before_resize = target_container_length - target_x_origin_relative_to_container - i + target_padding
				search_complete = True
				break
	# Search through the target container to find bottommost target pixel and calculate the height of the target
	target_height_before_resize = -1
	search_complete = False
	for i in range(target_container_height):
		if search_complete:
			break
		for j in range(target_container_length):
			if target_container_alpha_channel.getpixel((j, target_container_height-1-i)) > 10:
				target_height_before_resize = target_container_height - target_y_origin_relative_to_container - i + target_padding
				search_complete = True
				break
	target = target_container.crop((target_x_origin_relative_to_container, target_y_origin_relative_to_container, target_x_origin_relative_to_container+target_length_before_resize, target_y_origin_relative_to_container+target_height_before_resize))
	target_length = int(target_length_before_resize * target_size / max(target_length_before_resize, target_height_before_resize))
	target_height = int(target_height_before_resize * target_size / max(target_length_before_resize, target_height_before_resize))

	length_ratio = target_length / MAX_TARGET_SIZE
	height_ratio = target_height / MAX_TARGET_SIZE
	max_ratio = np.max([length_ratio, height_ratio])

	proportion_of_max = (13.0/15) + (2.0/15)*max_ratio

	rescale = proportion_of_max * MAX_TARGET_SIZE / np.max([target_length, target_height])
	target_length = int(target_length * rescale)
	target_height = int(target_height * rescale)



	randomized_target_params["target_length"] = target_length
	randomized_target_params["target_height"] = target_height
	target = target.resize((target_length, target_height))
	'''
	Step 5: Add gaussian noise to the target and blur
		1) Convert target to a raw numpy array
		2) Generate a noise array
		3) Add the noise array to the raw target array
		4) Convert back to pillow image
		5) Blur the image
	'''


	# Gaussian Blur

	target_raw = np.asarray(target)

	noise = np.zeros([target_height, target_length, 4])
	noise[:,:,0] = np.random.normal(noise_distribution_mean_1, noise_distribution_stddev_1, target_height*target_length).reshape(target_height,target_length).round()
	noise[:,:,1] = np.random.normal(noise_distribution_mean_2, noise_distribution_stddev_2, target_height*target_length).reshape(target_height,target_length).round()
	noise[:,:,2] = np.random.normal(noise_distribution_mean_3, noise_distribution_stddev_3, target_height*target_length).reshape(target_height,target_length).round()
	target_raw = (target_raw + 0.1*noise).clip(0,255).astype(np.uint8)

	# Apply 50% shading
	target = lighting_gradient(target_raw, target_length, target_height, 0.5)

	temp = np.asarray(target)
	temp = cv2.GaussianBlur(temp,(9,9),0)



	# Apply a perspective warp
	temp = transform(temp, 10)

	target = Image.fromarray(temp)

	return target


def draw_target_onto_background(target, background, randomized_target_params):
	target_x_offset = randomized_target_params["target_x_offset"]
	target_y_offset = randomized_target_params["target_y_offset"]
	target_length = randomized_target_params["target_length"]
	target_height = randomized_target_params["target_height"]
	# target_x_offset = random.choice(range(TARGET_X_OFFSET_MINMAX[0], IMAGE_LENGTH-target_length))
	# target_y_offset = random.choice(range(TARGET_Y_OFFSET_MINMAX[0], IMAGE_HEIGHT-target_height))
	target_x_offset = int(np.random.normal(0, IMAGE_LENGTH*TARGET_OFFSET_SPREAD))
	target_y_offset = int(np.random.normal(0, IMAGE_HEIGHT*TARGET_OFFSET_SPREAD))
	randomized_target_params["target_x_offset"] = target_x_offset
	randomized_target_params["target_y_offset"] = target_y_offset
	# Paste the target onto the background at the correct x,y offset. Also specify the mask so that the transparent areas don't get pasted
	background.paste(target, ((IMAGE_LENGTH-target_length)//2 + target_x_offset, (IMAGE_HEIGHT-target_height)//2 + target_y_offset), mask=target)

	target_raw = np.asarray(background)
	target_raw = cv2.GaussianBlur(target_raw,(11,11),0)
	background = Image.fromarray(target_raw)

def transform(image, amount):
	pts1 = np.float32([[amount,amount],[amount,150-amount],[150-amount,amount],[150-amount,150-amount]])
	pts2 = np.float32([[amount+ random.randint(-amount, amount), amount + random.randint(-amount, amount)],
		[amount+ random.randint(-amount, amount), 150 - amount + random.randint(-amount, amount)],
		[150 - amount + random.randint(-amount, amount), amount+ random.randint(-amount, amount)],
		[150 - amount + random.randint(-amount, amount) ,150 - amount + random.randint(-amount, amount)]])

	M = cv2.getPerspectiveTransform(pts1,pts2)
	image = cv2.warpPerspective(image,M, (150,150))
	return image

def save_image_and_annotations(image, randomized_image_params, randomized_target_params, img_savepath, label_savepath):
	# Convert the image to RGB just in case it's currently RGBA
	image = image.convert("RGB")
	# Save the image
	image.save(img_savepath)
	labels = {
		"shape": randomized_target_params["shape_class_dir"],
		"alphanumeric": randomized_target_params["alphanumeric"],
		"shape_color": randomized_target_params["shape_color_class_dir"],
		"alphanumeric_color": randomized_target_params["alphanumeric_color_class_dir"],
		"target_x_origin": randomized_target_params["target_x_offset"],
		"target_y_origin": randomized_target_params["target_y_offset"],
		"target_length": randomized_target_params["target_length"],
		"target_height": randomized_target_params["target_height"],
		"rotation_angle": randomized_target_params["rotation_angle"],
		"shape_filename": randomized_target_params["shape_filename"],
		"shape_color_filename": randomized_target_params["shape_color_filename"],
		"alphanumeric_color_filename": randomized_target_params["alphanumeric_color_filename"],
		"font_filename": randomized_target_params["font_filename"],
		"font_size": randomized_target_params["font_size"],
		"alphanumeric_offset_from_center_x": randomized_target_params["alphanumeric_offset_from_center_x"],
		"alphanumeric_offset_from_center_y": randomized_target_params["alphanumeric_offset_from_center_y"],
		"target_size": randomized_target_params["target_size"],
		"noise_distribution_mean_1": randomized_target_params["noise_distribution_mean_1"],
		"noise_distribution_stddev_1": randomized_target_params["noise_distribution_stddev_1"],
		"noise_distribution_mean_2": randomized_target_params["noise_distribution_mean_2"],
		"noise_distribution_stddev_2": randomized_target_params["noise_distribution_stddev_2"],
		"noise_distribution_mean_3": randomized_target_params["noise_distribution_mean_3"],
		"noise_distribution_stddev_3": randomized_target_params["noise_distribution_stddev_3"],
		"background_filename": randomized_image_params["background_filename"],
		"background_crop_vertex_left": randomized_image_params["background_crop_vertex_left"],
		"background_crop_vertex_top": randomized_image_params["background_crop_vertex_top"],
		"background_rotation_angle": randomized_image_params["background_rotation_angle"],
		"background_brightness_multiplier": randomized_image_params["background_brightness_multiplier"],
		"xmin": randomized_target_params["target_x_offset"],
		"xmax": randomized_target_params["target_x_offset"] + randomized_target_params["target_length"],
		"ymin": randomized_target_params["target_y_offset"],
		"ymax": randomized_target_params["target_y_offset"] + randomized_target_params["target_height"],
		"target_x_offset": randomized_target_params["target_x_offset"],
		"target_y_offset": randomized_target_params["target_y_offset"],
		"target_offset_spread": TARGET_OFFSET_SPREAD,
		"shadow_offset": SHADOW_OFFSET,
		"shadow_spread": SHADOW_SPREAD,
		"shadow_direction": randomized_target_params["shadow_direction"]
	}
	# write to the file
	json.dump(labels, open(label_savepath, 'w'), indent=1)




def generate_image(img_savepath, label_savepath, shape_class_dir, alphanumeric):
	randomized_image_params = choose_random_image_parameters()
	randomized_target_params = choose_random_target_parameters(shape_class_dir=shape_class_dir, alphanumeric=alphanumeric)
	background = generate_background(randomized_image_params)
	target = generate_target(randomized_target_params)
	draw_target_onto_background(target, background, randomized_target_params)

	image = background

	image = np.asarray(image)
	#image = cv2.resize(image, (299, 299))


	image = Image.fromarray(image)


	save_image_and_annotations(image, randomized_image_params, randomized_target_params, img_savepath, label_savepath)
	target.close()
	image.close()




if __name__ == '__main__':

	# Get current time in seconds since epoch, will use this to generate filenames
	timestamp = str(int(time.time()))


	shape_class_dirs = [folder for folder in os.listdir(SHAPE_DIR) if not folder.startswith(".")]
	color_choices = os.listdir(COLOR_DIR)

	num_classes = len(shape_class_dirs) * len(ALPHANUMERIC_CHOICES)
	num_images_to_generate = NUM_IMAGES_PER_CLASS_TO_GENERATE * num_classes


	# Write to the annotations file
	annotations = {}
	annotations["shape"] = {shape_class_dirs[i]: i+1 for i in range(len(shape_class_dirs))}
	annotations["alphanumeric"] = {ALPHANUMERIC_CHOICES[i]: i for i in range(len(ALPHANUMERIC_CHOICES))}
	annotations["shape_color"] = {color_choices[i]: i for i in range(len(color_choices))}
	annotations["alphanumeric_color"] = {color_choices[i]: i for i in range(len(color_choices))}
	json.dump(annotations, open(ANNOTATIONS_FILEPATH, 'w'), indent=1)

	# Variables used for storing and reloading from the state file
	num_images_skipped = 0
	num_images_to_skip = 0
	if CONTINUE_FROM_SAVED_STATE == True:
		saved_state = json.load(open(STATE_FILEPATH, 'r'))
		num_images_to_skip = saved_state["generated"]
		num_images_to_generate = saved_state["total"]
		NUM_IMAGES_PER_CLASS_TO_GENERATE = int(num_images_to_generate / num_classes)
		timestamp = saved_state["timestamp"]

	# Print some useful stuff for the user
	print("\nGenerating " + str(NUM_IMAGES_PER_CLASS_TO_GENERATE) + " images per class. Number of classes: " + str(num_classes) + ". Total images being generated: " + str(num_images_to_generate))
	if CONTINUE_FROM_SAVED_STATE == True:
		print("\nContinuing from where previous run left off, skipping " + str(num_images_to_skip) + " images\n")


	count = 0
	for shape_class_dir in shape_class_dirs:
		for alphanumeric in ALPHANUMERIC_CHOICES:
			for i in range(NUM_IMAGES_PER_CLASS_TO_GENERATE):
				# Skip images until we are up to where we left off in the previous run (images are only skipped if the user specifies the continue flag as true)
				if num_images_skipped != num_images_to_skip:
					num_images_skipped += 1
					count += 1
					continue
				print("Generating image #" + str(count+1) + "/" + str(num_images_to_generate))
				# Generate the save filepath for the image and json file using the system time and count
				img_savepath = PIC_SAVE_DIR + timestamp + "-" + str(count+1) + ".jpg"
				label_savepath = LABEL_SAVE_DIR + timestamp + "-" + str(count+1) + ".json"
				# Generate the image.
				generate_image(img_savepath, label_savepath, shape_class_dir, alphanumeric)
				# Dump the progress information to the state_filename
				state = {"total": num_images_to_generate, "generated": count+1, "timestamp": timestamp}
				json.dump(state, open(STATE_FILEPATH, 'w'))
				count += 1
