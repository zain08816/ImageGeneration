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

parser = argparse.ArgumentParser()
parser.add_argument('--num_images',default=100,type=int,help='num images to generate')
parser.add_argument('--image_length',default=150,type=int,help='Length dimension of output images')
parser.add_argument('--image_height',default=150,type=int,help='Height dimension of output images')
args = parser.parse_args()

NUM_IMAGES_TO_GENERATE = args.num_images
IMAGE_LENGTH = args.image_length
IMAGE_HEIGHT = args.image_height



PARENT_SAVE_DIR = "generated-for-tesseract/"
PIC_SAVE_DIR = PARENT_SAVE_DIR + "images/"
LABEL_SAVE_DIR = PARENT_SAVE_DIR + "labels/"


# If the following directories don't exist, create them
if not os.path.exists(PARENT_SAVE_DIR):
	os.mkdir(PARENT_SAVE_DIR)
if not os.path.exists(PIC_SAVE_DIR):
	os.mkdir(PIC_SAVE_DIR)
if not os.path.exists(LABEL_SAVE_DIR):
	os.mkdir(LABEL_SAVE_DIR)

	

ALPHANUMERIC_CHOICES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', 'a', 'A', 'b', 'B', 'c', 'C', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'j', 'J', 'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 't' ,'T', 'u', 'U', 'v', 'V', 'x', 'X', 'y', 'Y', 'z', 'Z']
FONT_DIR = "Fonts/"
FONT_SIZE_RANGE = range(50, 100)
ALPHANUMERIC_OFFSET_FROM_CENTER_RANGE = range(-3, 4)
ROTATION_ANGLE_RANGE = range(0, 360)



def generate_image(img_savepath, label_savepath):

	# Choose the randomized parameters
	alphanumeric = random.choice(ALPHANUMERIC_CHOICES)
	font_filename = random.choice(os.listdir(FONT_DIR))
	font_size = random.choice(FONT_SIZE_RANGE)
	alphanumeric_offset_from_center_x = random.choice(ALPHANUMERIC_OFFSET_FROM_CENTER_RANGE)
	alphanumeric_offset_from_center_y = random.choice(ALPHANUMERIC_OFFSET_FROM_CENTER_RANGE)
	rotation_angle = random.choice(ROTATION_ANGLE_RANGE)
	
	background_rgba = (255,255,255,255)
	alphanumeric_rgba = (0,0,0,255)
	
	# Create the white background
	image = Image.new(mode="RGBA", size=(IMAGE_LENGTH, IMAGE_HEIGHT), color=background_rgba)
	
	# Draw the alphanumeric onto the background
	font = ImageFont.truetype(FONT_DIR+font_filename, font_size)
	alphanumeric_width_including_whitespace, alphanumeric_height_including_whitespace = font.getsize(alphanumeric)
	alphanumeric_whitespace_x, alphanumeric_whitespace_y = font.getoffset(alphanumeric)
	alphanumeric_width = alphanumeric_width_including_whitespace - alphanumeric_whitespace_x
	alphanumeric_height = alphanumeric_height_including_whitespace - alphanumeric_whitespace_y
	draw_position_x = (image.width / 2) - alphanumeric_whitespace_x - (alphanumeric_width / 2) + alphanumeric_offset_from_center_x
	draw_position_y = (image.height / 2) - alphanumeric_whitespace_y - (alphanumeric_height / 2) + alphanumeric_offset_from_center_y
	drawer = ImageDraw.Draw(image)
	drawer.text((draw_position_x, draw_position_y), alphanumeric, fill=tuple(alphanumeric_rgba), font=font)
	
	# Rotate and resize the image. Pillow adds padding with incorrect colors to rotations so we have to paste back the correct color 
	image = image.rotate(rotation_angle, expand=True)
	image = image.resize((IMAGE_LENGTH, IMAGE_HEIGHT))
	black_filter = Image.new("RGBA", image.size, background_rgba)
	image = Image.composite(image, black_filter, image)
	
	
	# Save the image and labels
	image.convert("RGB").save(img_savepath)
	labels = {
		"alphanumeric_xmin": draw_position_x,
		"alphanumeric_xmax": draw_position_x + alphanumeric_width,
		"alphanumeric_ymin": draw_position_y,
		"alphanumeric_ymax": draw_position_y + alphanumeric_height,
		"alphanumeric": alphanumeric,
		"alphanumeric_color": alphanumeric_rgba,
		"background_color": background_rgba,
		"rotation_angle": rotation_angle,
		"font_filename": font_filename,
		"font_size": font_size,
		"alphanumeric_offset_from_center_x": alphanumeric_offset_from_center_x,
		"alphanumeric_offset_from_center_y": alphanumeric_offset_from_center_y,
	}
	json.dump(labels, open(label_savepath, 'w'), indent=1)
	
	image.close()
	
	


if __name__ == '__main__':

	# Get current time in seconds since epoch, will use this to generate filenames
	timestamp = str(int(time.time()))

	count = 0
	for i in range(NUM_IMAGES_TO_GENERATE):
		print("Generating image #" + str(count+1) + "/" + str(NUM_IMAGES_TO_GENERATE))
		# Generate the save filepath for the image and json file using the system time and count
		img_savepath = PIC_SAVE_DIR + timestamp + "-" + str(count+1) + ".jpg"
		label_savepath = LABEL_SAVE_DIR + timestamp + "-" + str(count+1) + ".json"
		# Generate the image.
		generate_image(img_savepath, label_savepath)
		count += 1
