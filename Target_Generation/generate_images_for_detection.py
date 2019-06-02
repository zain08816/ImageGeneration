import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import numpy as np
import json
import pdb
from multiprocessing import Process
import argparse
import time
import sys
from generate_images_for_classification_for1819 import choose_random_target_parameters, generate_target

import threading

parser = argparse.ArgumentParser()
parser.add_argument('--num_pics',default=1000,type=int,help='num images to generate')
parser.add_argument('--image_length',default=1000,type=int,help='Length dimension of output images')
parser.add_argument('--image_height',default=600,type=int,help='Height dimension of output images')
parser.add_argument('--datadir',default="generated-for-detection",type=str,help='Generated Image dir')
parser.add_argument('--bigger_targets_bool', default=False, type=bool, help='If True, targets will be made significantly bigger')
parser.add_argument('--enable_multi_task', default=False, type=bool, help='If True, annotations will be formatted for multi-task detector.')
parser.add_argument('--uppercase_only_bool', default = True, type=bool,help='If True, only capital letters will be generated')
parser.add_argument('--alpha_only', default=False, type=bool, help='If True, only capital alphabets')
args = parser.parse_args()
print(args.uppercase_only_bool)
NUM_IMAGES_TO_GENERATE = args.num_pics
IMAGE_LENGTH = args.image_length
IMAGE_HEIGHT = args.image_height
CAPITAL_ONLY = args.uppercase_only_bool
BIGGER_BOOl = args.bigger_targets_bool

PARENT_SAVE_DIR = args.datadir+"/"
PIC_SAVE_DIR = PARENT_SAVE_DIR + "images/"
LABEL_SAVE_DIR = PARENT_SAVE_DIR + "labels/"
ANNOTATIONS_FILEPATH = PARENT_SAVE_DIR + "annotations.json"
STATE_FILEPATH = PARENT_SAVE_DIR + ".state.json"
MULTI_TASK = args.enable_multi_task
ALPHA_ONLY = args.alpha_only
# If the following directories don't exist, create them
if not os.path.exists(PARENT_SAVE_DIR):
	os.mkdir(PARENT_SAVE_DIR)
if not os.path.exists(PIC_SAVE_DIR):
	os.mkdir(PIC_SAVE_DIR)
if not os.path.exists(LABEL_SAVE_DIR):
	os.mkdir(LABEL_SAVE_DIR)




'''
HYPERPARAMETERS FOR RANDOMIZED PARAMETERS DEFINED HERE
Every time this script generates an image, it randomizes several parameters, such as rotation angle, size, font, etc.
The hyperparameters constraining the choices/ranges/directory-paths for these parameters are defined here. Feel free to edit the hyperparameters.
Make sure you read the comments before editing any hyperparameters, otherwise you might break something
'''
SHAPE_DIR = "Shapes/"
ALPHANUMERIC_CHOICES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', 'a', 'A', 'b', 'B', 'c', 'C', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'j', 'J', 'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 't' ,'T', 'u', 'U', 'v', 'V', 'x', 'X', 'y', 'Y', 'z', 'Z']
if (CAPITAL_ONLY):
	ALPHANUMERIC_CHOICES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
if (ALPHA_ONLY):
	ALPHANUMERIC_CHOICES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
print(ALPHANUMERIC_CHOICES)
COLOR_DIR = "Colors/"
FONT_DIR = "Fonts_New/Fonts/"
FONT_SIZE_MINMAX = [6, None]
ALPHANUMERIC_OFFSET_FROM_CENTER_RANGE = range(-1, 2)
ROTATION_ANGLE_RANGE = range(0, 360)
TARGET_SIZE_RANGE = range(30,40) if not BIGGER_BOOl else range(20, 30)



NOISE_DISTRIBUTION_MEAN_RANGE = range(-20, 2)
NOISE_DISTRIBUTION_STDDEV_RANGE = range(0, 1)
BACKGROUND_DIR = "Backgrounds/"
BACKGROUND_CROP_VERTEX_LEFT_RANGE = None
BACKGROUND_CROP_VERTEX_TOP_RANGE = None
BACKGROUND_ROTATION_ANGLE_CHOICES = [0, 180]
BACKGROUND_BRIGHTNESS_MULTIPLIER_RANGE = np.arange(0.7, 1, 0.8)
TARGET_X_OFFSET_MINMAX = [0, None]
TARGET_Y_OFFSET_MINMAX = [0, None]
NUM_TARGETS_IN_IMAGE_RANGE = range(3, 5)

#These colors are difficult to distinguish. We are not holding our neural network responsible for these.

illegal_color_combinations = [ #(shape_color, text_color)
	["white", "gray"],
	["gray", "white"],
	["gray", "brown"],
	["brown", "gray"],
	["green", "white"]
	]

num_pixels_in_image = IMAGE_LENGTH * IMAGE_HEIGHT
num_pixels_in_target_max = TARGET_SIZE_RANGE[-1] * TARGET_SIZE_RANGE[-1]
if num_pixels_in_target_max > num_pixels_in_image:
	print("!!!!!!!!!!!!!Error: target will not fit in image, either make target smaller or image larger!!!!!!!!!!!!!!!!!")
	sys.exit(1)




def choose_random_image_parameters():
	possible_background_files = [file for file in os.listdir(BACKGROUND_DIR) if not file.startswith(".")]
	background_filename = random.choice(possible_background_files)
	# Can't choose the left/top crop vertices here. Need to know the size of the full background image
	background_crop_vertex_left = None
	background_crop_vertex_top = None
	background_rotation_angle = random.choice(BACKGROUND_ROTATION_ANGLE_CHOICES)
	background_brightness_multiplier = random.choice(BACKGROUND_BRIGHTNESS_MULTIPLIER_RANGE)
	num_targets_in_image = random.choice(NUM_TARGETS_IN_IMAGE_RANGE)
	randomized_image_params = {
				"background_filename": background_filename,
				"background_crop_vertex_left": background_crop_vertex_left,
				"background_crop_vertex_top": background_crop_vertex_top,
				"background_rotation_angle": background_rotation_angle,
				"background_brightness_multiplier": background_brightness_multiplier,
				"num_targets_in_image": num_targets_in_image,
				}
	return randomized_image_params


def choose_random_target_parameters():
 	possible_shape_class_directories = [file for file in os.listdir(SHAPE_DIR) if not file.startswith(".")]
 	shape_class_dir = random.choice(possible_shape_class_directories)
 	shape_filename = random.choice(os.listdir(SHAPE_DIR + shape_class_dir + "/"))
 	shape_color_class_dir = random.choice(os.listdir(COLOR_DIR))
 	shape_color_filename = random.choice(os.listdir(COLOR_DIR + shape_color_class_dir + "/"))
 	alphanumeric = random.choice(ALPHANUMERIC_CHOICES)
 	alphanumeric_color_class_dir = random.choice(os.listdir(COLOR_DIR))
 	#If the colors are the same, or it is an illegal color combination, repick the colors.
 	while alphanumeric_color_class_dir == shape_color_class_dir or [shape_color_class_dir, alphanumeric_color_class_dir] in illegal_color_combinations:
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
 						}
 	return randomized_target_params

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
		1) Open the shape image
		2) Color it
	'''
	target_container = Image.open(SHAPE_DIR+shape_class_dir+"/"+shape_filename)
	with open(COLOR_DIR+shape_color_class_dir+"/"+shape_color_filename, 'r') as shape_color_file:
		shape_color = json.load(shape_color_file)
	shape_color.append(255)
	shape_color = tuple(shape_color)
	target_container.paste(shape_color, (0,0,target_container.size[0],target_container.size[1]), mask=target_container)
	'''
	Step 3: Draw the alphanumeric on the target container and execute the post-alphanumeric-draw operations:
		1) Calculate the maximum allowable font size and choose a font size
			a) Predict the aspect ratio of the alphanumeric
			b) Find the largest rectangle (with an spect ratio equal to the aspect ratio of the alphanumeric) that fits inside the shape
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
				alphanumeric_maxsize = (right-left-1, bottom-top-1)
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
		font_size = random.choice(range(max(FONT_SIZE_MINMAX[0],int(font_size_max/2)), font_size_max))
		randomized_target_params["font_size"] = font_size
	elif font_size == None and FONT_SIZE_MINMAX[1] != None:
		font_size = random.choice(range(max(FONT_SIZE_MINMAX[0],int(FONT_SIZE_MINMAX[1]/2)), FONT_SIZE_MINMAX[1]))
		randomized_target_params["font_size"] = font_size
	font = ImageFont.truetype(FONT_DIR+font_filename, font_size)
	alphanumeric_width_including_whitespace, alphanumeric_height_including_whitespace = font.getsize(alphanumeric)
	alphanumeric_whitespace_x, alphanumeric_whitespace_y = font.getoffset(alphanumeric)
	alphanumeric_width = alphanumeric_width_including_whitespace - alphanumeric_whitespace_x
	alphanumeric_height = alphanumeric_height_including_whitespace - alphanumeric_whitespace_y
	draw_position_x = (target_container.width / 2) - alphanumeric_whitespace_x - (alphanumeric_width / 2)
		# + alphanumeric_offset_from_center_x
	draw_position_y = (target_container.height / 2) - alphanumeric_whitespace_y - (alphanumeric_height / 2)
		# 	+ alphanumeric_offset_from_center_y

	#print(alphanumeric_offset_from_center_x,alphanumeric_offset_from_center_y)
	drawer = ImageDraw.Draw(target_container)
	with open(COLOR_DIR+alphanumeric_color_class_dir+"/"+alphanumeric_color_filename, 'r') as alphanumeric_color_file:
		alphanumeric_color_rgb = json.load(alphanumeric_color_file)
	drawer.text((draw_position_x, draw_position_y), alphanumeric, fill=tuple(alphanumeric_color_rgb), font=font)
	target_container = target_container.rotate(rotation_angle, expand=True)
	'''
	Step 4: Determine the target's dimensions and position relative to the target container (note: this is pretty stupid but I couldn't think of anything better) and use that information to choose the x and y offsets
		1) Search through the target container for the leftmost, topmost, rightmost, and bottommost pixels that are part of the target in order to determine the x-origin, y-origin, length, and height (this is pretty stupid that I have to do this)
		2) Crop out the target
		3) Resize the target
		4) Choose the x and y offsets (we can do this now because we know the final size of the target)


	'''

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
			if target_container_alpha_channel.getpixel((i, j)) != 0:
				target_x_origin_relative_to_container = i
				search_complete = True
				break
	# Search through the target container to determine the y-origin of the target relative to the target container
	target_y_origin_relative_to_container = -1
	search_complete = False
	for i in range(target_container_height):
		if search_complete:
			break
		for j in range(target_container_length):
			if target_container_alpha_channel.getpixel((j, i)) != 0:
				target_y_origin_relative_to_container = i
				search_complete = True
				break
	# Search through the target container to find rightmost target pixel and calculate the length of the target
	target_length_before_resize = -1
	search_complete = False
	for i in range(target_container_length):
		if search_complete:
			break
		for j in range(target_container_height):
			if target_container_alpha_channel.getpixel((target_container_length-1-i, j)) != 0:
				target_length_before_resize = target_container_length - target_x_origin_relative_to_container - i
				search_complete = True
				break
	# Search through the target container to find bottommost target pixel and calculate the height of the target
	target_height_before_resize = -1
	search_complete = False
	for i in range(target_container_height):
		if search_complete:
			break
		for j in range(target_container_length):
			if target_container_alpha_channel.getpixel((j, target_container_height-1-i)) != 0:
				target_height_before_resize = target_container_height - target_y_origin_relative_to_container - i
				search_complete = True
				break

	# Adjust size based on SHAPE_DIR
	if (shape_class_dir == "star"):
		target_size += 20
	if (shape_class_dir == "cross"):
		target_size += 10

	target = target_container.crop((target_x_origin_relative_to_container, target_y_origin_relative_to_container, target_x_origin_relative_to_container+target_length_before_resize, target_y_origin_relative_to_container+target_height_before_resize))
	target_length = int(target_length_before_resize * target_size / max(target_length_before_resize, target_height_before_resize))
	target_height = int(target_height_before_resize * target_size / max(target_length_before_resize, target_height_before_resize))
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
	'''
	target_raw = np.asarray(target)
	noise = np.zeros([target_height, target_length, 4])
	noise[:,:,0] = np.random.normal(noise_distribution_mean_1, noise_distribution_stddev_1, target_height*target_length).reshape(target_height,target_length).round()
	noise[:,:,1] = np.random.normal(noise_distribution_mean_2, noise_distribution_stddev_2, target_height*target_length).reshape(target_height,target_length).round()
	noise[:,:,2] = np.random.normal(noise_distribution_mean_3, noise_distribution_stddev_3, target_height*target_length).reshape(target_height,target_length).round()
	target_raw = (target_raw + noise).clip(0,255).astype(np.uint8)
	target = Image.fromarray(target_raw)
	target = target.filter(ImageFilter.MedianFilter(size=3))
		'''
	return target

def draw_target_onto_background(target, background, randomized_target_params, randomized_target_params_list):
	target_x_offset = randomized_target_params["target_x_offset"]
	target_y_offset = randomized_target_params["target_y_offset"]
	target_length = randomized_target_params["target_length"]
	target_height = randomized_target_params["target_height"]
	previous_target_x_offsets = [d["target_x_offset"] for d in randomized_target_params_list]
	previous_target_y_offsets = [d["target_y_offset"] for d in randomized_target_params_list]
	previous_target_lengths = [d["target_length"] for d in randomized_target_params_list]
	previous_target_heights = [d["target_height"] for d in randomized_target_params_list]
	count = 0
	while(True):
		if count == 50:
			print("!!!!!!!!!!!!!!Warning: Cant seem to fit target on image, skipping!!!!!!!!!!!!!!!")
			return False
		target_x_offset = random.choice(range(TARGET_X_OFFSET_MINMAX[0], IMAGE_LENGTH-target_length))
		target_y_offset = random.choice(range(TARGET_Y_OFFSET_MINMAX[0], IMAGE_HEIGHT-target_height))
		xmin = target_x_offset
		xmax = target_x_offset + target_length
		ymin = target_y_offset
		ymax = target_y_offset + target_height
		overlaps = False
		for i in range(len(previous_target_x_offsets)):
			prev_xmin = previous_target_x_offsets[i]
			prev_xmax = previous_target_x_offsets[i] + previous_target_lengths[i]
			prev_ymin = previous_target_y_offsets[i]
			prev_ymax = previous_target_y_offsets[i] + previous_target_heights[i]
			if (((xmin>=prev_xmin and xmin<=prev_xmax) or (xmax>=prev_xmin and xmax<=prev_xmax) or (xmin<=prev_xmin and xmax>=prev_xmax)) and ((ymin>=prev_ymin and ymin<=prev_ymax) or (ymax>=prev_ymin and ymax<=prev_ymax) or (ymin<=prev_ymin and ymax>=prev_ymax))):
				overlaps = True
		if overlaps == True:
			count += 1
			continue
		randomized_target_params["target_x_offset"] = target_x_offset
		randomized_target_params["target_y_offset"] = target_y_offset
		break
	# Paste the target onto the background at the correct x,y offset. Also specify the mask so that the transparent areas don't get pasted
	background.paste(target, (target_x_offset, target_y_offset), mask=target)

def save_image_and_annotations(image, randomized_image_params, randomized_target_params_list, img_savepath, label_savepath):
	# Convert the image to RGB just in case it's currently RGBA
	image = image.convert("RGB")
	# Save the image
	image.save(img_savepath)
	if not MULTI_TASK:
		labels = {
			"image/height" : [IMAGE_HEIGHT,],
			"image/width" : [IMAGE_LENGTH,],
			"image/filename" : img_savepath,
			"image/object/bbox/xmin" : [],
			"image/object/bbox/xmax" : [],
			"image/object/bbox/ymin" : [],
			"image/object/bbox/ymax" : [],
			"image/object/class/text" : [],
			"image/object/class/label": [],

			"shape": [],
			"alphanumeric": [],
			"shape_color": [],
			"alphanumeric_color": [],
			"target_x_origin": [],
			"target_y_origin": [],
			"target_length": [],
			"target_height": [],
			"rotation_angle": [],
			"shape_filename": [],
			"shape_color_filename": [],
			"alphanumeric_color_filename": [],
			"font_filename": [],
			"font_size": [],
			"alphanumeric_offset_from_center_x": [],
			"alphanumeric_offset_from_center_y": [],
			"target_size": [],
			"noise_distribution_mean_1": [],
			"noise_distribution_stddev_1": [],
			"noise_distribution_mean_2": [],
			"noise_distribution_stddev_2": [],
			"noise_distribution_mean_3": [],
			"noise_distribution_stddev_3": [],
			"background_filename": randomized_image_params["background_filename"],
			"background_crop_vertex_left": randomized_image_params["background_crop_vertex_left"],
			"background_crop_vertex_top": randomized_image_params["background_crop_vertex_top"],
			"background_rotation_angle": randomized_image_params["background_rotation_angle"],
			"background_brightness_multiplier": randomized_image_params["background_brightness_multiplier"],
			"num_targets_in_image": randomized_image_params["num_targets_in_image"],
			# "xmin" : [],
			# "xmax" : [],
			# "ymin" : [],
			# "ymax" : []
		}
	else:
		labels = {
			"image/height" : [IMAGE_HEIGHT,],
			"image/width" : [IMAGE_LENGTH,],
			# "image/filename" : img_savepath,
			"image/object/bbox/xmin" : [],
			"image/object/bbox/xmax" : [],
			"image/object/bbox/ymin" : [],
			"image/object/bbox/ymax" : [],
			"image/object/class/shape" : [],
			"image/object/class/alphanumeric" : [],
			"shape": [],
			"alphanumeric": [],
			"shape_color": [],
			"alphanumeric_color": [],
			"target_x_origin": [],
			"target_y_origin": [],
			"target_length": [],
			"target_height": [],
			"rotation_angle": [],
			"shape_filename": [],
			"shape_color_filename": [],
			"alphanumeric_color_filename": [],
			"font_filename": [],
			"font_size": [],
			"alphanumeric_offset_from_center_x": [],
			"alphanumeric_offset_from_center_y": [],
			"target_size": [],
			"noise_distribution_mean_1": [],
			"noise_distribution_stddev_1": [],
			"noise_distribution_mean_2": [],
			"noise_distribution_stddev_2": [],
			"noise_distribution_mean_3": [],
			"noise_distribution_stddev_3": [],
			"background_filename": randomized_image_params["background_filename"],
			"background_crop_vertex_left": randomized_image_params["background_crop_vertex_left"],
			"background_crop_vertex_top": randomized_image_params["background_crop_vertex_top"],
			"background_rotation_angle": randomized_image_params["background_rotation_angle"],
			"background_brightness_multiplier": randomized_image_params["background_brightness_multiplier"],
			"num_targets_in_image": randomized_image_params["num_targets_in_image"],
			# "xmin" : [],
			# "xmax" : [],
			# "ymin" : [],
			# "ymax" : []
		}
	for randomized_target_params in randomized_target_params_list:
	# 	'''
	# 	Unpack the randomized target params
	# 	'''
	#
		shapeDictionary = {shape_class_dirs[i]: i+1 for i in range(len(shape_class_dirs))}
		alphanumericDictionary = {c: i+1 for i,c in enumerate(ALPHANUMERIC_CHOICES)}
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
		target_x_offset = randomized_target_params["target_x_offset"]
		target_y_offset = randomized_target_params["target_y_offset"]
		target_length = randomized_target_params["target_length"]
		target_height = randomized_target_params["target_height"]
	#
		xmin = target_x_offset
		xmax = target_x_offset + target_length
		ymin = target_y_offset
		ymax = target_y_offset + target_height
	#
	# 	'''Write to the labels file'''
		labels["shape"].append(shape_class_dir)
		labels["alphanumeric"].append(alphanumeric)
		labels["shape_color"].append(shape_color_class_dir)
		labels["alphanumeric_color"].append(alphanumeric_color_class_dir)
		labels["target_x_origin"].append(target_x_offset)
		labels["target_y_origin"].append(target_y_offset)
		labels["target_length"].append(target_length)
		labels["target_height"].append(target_height)
		labels["rotation_angle"].append(rotation_angle)
		labels["shape_filename"].append(shape_filename)
		labels["shape_color_filename"].append(shape_color_filename)
		labels["alphanumeric_color_filename"].append(alphanumeric_color_filename)
		labels["font_filename"].append(font_filename)
		labels["font_size"].append(font_size)
		labels["alphanumeric_offset_from_center_x"].append(alphanumeric_offset_from_center_x)
		labels["alphanumeric_offset_from_center_y"].append(alphanumeric_offset_from_center_y)
		labels["target_size"].append(target_size)
		labels["noise_distribution_mean_1"].append(noise_distribution_mean_1)
		labels["noise_distribution_stddev_1"].append(noise_distribution_stddev_1)
		labels["noise_distribution_mean_2"].append(noise_distribution_mean_2)
		labels["noise_distribution_stddev_2"].append(noise_distribution_stddev_2)
		labels["noise_distribution_mean_3"].append(noise_distribution_mean_3)
		labels["noise_distribution_stddev_3"].append(noise_distribution_stddev_3)
		if not MULTI_TASK:
			labels["image/object/class/text"].append(shapeDictionary[shape_class_dir])
			labels["image/object/class/label"].append(shapeDictionary[shape_class_dir])
			labels["image/object/bbox/xmin"].append(xmin)
			labels["image/object/bbox/xmax"].append(xmax)
			labels["image/object/bbox/ymin"].append(ymin)
			labels["image/object/bbox/ymax"].append(ymax)
		else:
			labels['image/object/class/shape'].append(shapeDictionary[shape_class_dir])
			labels['image/object/class/alphanumeric'].append(alphanumericDictionary[alphanumeric])
			labels["image/object/class/text"].append(shapeDictionary[shape_class_dir])
			labels["image/object/class/label"].append(shapeDictionary[shape_class_dir])
			labels["image/object/bbox/xmin"].append(xmin)
			labels["image/object/bbox/xmax"].append(xmax)
			labels["image/object/bbox/ymin"].append(ymin)
			labels["image/object/bbox/ymax"].append(ymax)

	json.dump(labels, open(label_savepath, 'w'), indent=1)




def generate_image(img_savepath, label_savepath):
	randomized_image_params = choose_random_image_parameters()
	background = generate_background(randomized_image_params)
	num_targets_in_image = randomized_image_params["num_targets_in_image"]
	randomized_target_params_list = []
	for i in range(num_targets_in_image):
		randomized_target_params = choose_random_target_parameters()
		target = generate_target(randomized_target_params)
		result = draw_target_onto_background(target, background, randomized_target_params, randomized_target_params_list)
		target.close()
		if result != False:
			randomized_target_params_list.append(randomized_target_params)
	image = background
	save_image_and_annotations(image, randomized_image_params, randomized_target_params_list, img_savepath, label_savepath)
	image.close()




if __name__ == '__main__':

	# Get current time in seconds since epoch, will use this to generate filenames
	timestamp = str(int(time.time()))


	shape_class_dirs = os.listdir(SHAPE_DIR)
	color_choices = os.listdir(COLOR_DIR)


	# Write to the annotations file
	annotations = {}
	annotations["shape"] = {shape_class_dirs[i]: i+1 for i in range(len(shape_class_dirs))}
	annotations["alphanumeric"] = {ALPHANUMERIC_CHOICES[i]: i for i in range(len(ALPHANUMERIC_CHOICES))}
	annotations["shape_color"] = {color_choices[i]: i for i in range(len(color_choices))}
	annotations["alphanumeric_color"] = {color_choices[i]: i for i in range(len(color_choices))}


	annotations["image/object/class/text"] = {shape_class_dirs[i]: i+1 for i in range(len(shape_class_dirs))}
	annotations["image/object/class/label"] = {shape_class_dirs[i]: i+1 for i in range(len(shape_class_dirs))}
	annotations["image/filename"] = 'str'
	annotations["image/width"] = 'int'
	annotations["image/height"] = 'int'


	annotations["image/object/bbox/xmin"] = 'float'
	annotations["image/object/bbox/xmax"] = 'float'
	annotations["image/object/bbox/ymin"] = 'float'
	annotations["image/object/bbox/ymax"] = 'float'

	json.dump(annotations, open(ANNOTATIONS_FILEPATH, 'w'), indent=1)


	# Print some useful stuff for the user
	print("Total images being generated: " + str(NUM_IMAGES_TO_GENERATE))



	# Keep track of the number of images generated

	threads = []



	count = 0
	# for i in range(NUM_IMAGES_TO_GENERATE):
	# 	print("Generating image #" + str(count+1) + "/" + str(NUM_IMAGES_TO_GENERATE))
	# 	# Generate the save filepath for the image and json file using the system time and count
	# 	img_savepath = PIC_SAVE_DIR + timestamp + "-" + str(count+1) + ".jpg"
	# 	label_savepath = LABEL_SAVE_DIR + timestamp + "-" + str(count+1) + ".json"
	# 	# Generate the image. Pass in all the savepaths, iterated parameters, and randomized parameters
	# 	generate_image(img_savepath, label_savepath)
	# 	count += 1

	total = int(NUM_IMAGES_TO_GENERATE/10)

	for i in range(total):
		for z in range(10):
			print("Generating image #" + str(count+1) + "/" + str(NUM_IMAGES_TO_GENERATE))
			img_savepath = PIC_SAVE_DIR + timestamp + "-" + str(count+1) + ".jpg"
			label_savepath = LABEL_SAVE_DIR + timestamp + "-" + str(count+1) + ".json"
			thread = threading.Thread(target=generate_image, args=(img_savepath,label_savepath))
			threads.append(thread)
			thread.start()
			count += 1

		for x in threads:
			x.join()
		theards = []