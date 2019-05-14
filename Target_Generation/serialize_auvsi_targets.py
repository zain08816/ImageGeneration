import sys
import tarfile
from six.moves import urllib
import os
import numpy as np
from PIL import Image
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--serial_out',default='target_serial',help='output file name')
parser.add_argument('--dataset',default='target_images',help='directory containing "train" and "test" images')
parser.add_argument('--img_per_bin',type=int,default=10000)
#parser.add_argument('--training',type=bool,default=False)



args = parser.parse_args()

dataset = args.dataset
#training = args.training

if not os.path.exists(dataset):
	raise Exception('%s does not exist' %dataset)
serial_out = args.serial_out

if  os.path.exists(serial_out):
	raise Exception('%s does exists ' %serial_out )
os.mkdir(serial_out)
"""
if training and os.path.exists(os.path.join(serial_out,'train')):
	raise Exception('"train" dir in %s already exists...' %serial_out)
if not training and os.path.exists(os.path.join(serial_out,'test')):
	raise Exception('"test" dir in %s already exists...' % serial_out)

if training:
	train_path = os.path.join(serial_out,'train')
	os.mkdir(train_path)
	serial_out = train_path
else:
	test_path = os.path.join(serial_out,'test')
	os.mkdir(test_path)
	serial_out = test_path
"""
root_path = dataset
img_per_bin = args.img_per_bin




shapes = ['circle','cross','heptagon','hexagon','octagon','pentagon','quartercircle',
'rectangle','semicircle','square','star','trapezoid','triangle']
colors = ['black','blue','brown','gray','green','orange','purple','red','white','yellow']
#chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'A', 'b', 'B', 'c', 'C', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'j', 'J', 'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 't' ,'T', 'u', 'U', 'v', 'V', 'w', 'W', 'x', 'X', 'y', 'Y', 'z', 'Z']
chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', 'a', 'A', 'b', 'B', 'c', 'C', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'j', 'J', 'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 't' ,'T', 'u', 'U', 'v', 'V', 'x', 'X', 'y', 'Y', 'z', 'Z']




shape_num = {c:i for i,c in enumerate(shapes)}
color_num = {c:i for i,c in enumerate(colors)}
chars     = {c:i for i,c in enumerate(chars)}


print(root_path)

file_names = []

j = 0
num = 0
f = None

#shuffle targets in directory
for file in os.listdir(root_path):
	file_names.append(file)
for _ in range(60):
	random.shuffle(file_names)

"""
a = [ x.split('-')[0] for x in file_names[:200]]
for b in classes.keys():
	print(b,a.count(b))
"""
for file in file_names:
	if num %img_per_bin == 0:
		if f is not None:
			f.close()
		f = open(os.path.join(serial_out,('auvsi_train%d.bin') %j),"ba")
		j+=1
	num+=1
	image= os.path.join(root_path,file)
	if file.endswith('.jpg'):
		class_names = file.split('_')
		shape =  class_names[0].split('-')[0]
		char = class_names[1].split('-')[0]
		char_color = class_names[2].split('-')[0]
		bkgn_color       = class_names[3]
		print(shape,char)
		if shape not in shape_num:
			raise Exception(shape+'not in classes')
		if bkgn_color not in color_num:
			raise Exception(bkgn_color +'not in classes')
		if char_color not in color_num:
			raise Exception(char_color+'not in classes')
		if char not in chars:
			raise Exception(char+' not in classes')
		print(shape,bkgn_color,char_color,char)
		img = Image.open(image)
		#img = img.resize((299,299),Image.ANTIALIAS)
		img = np.array(img)
		img_dim = np.shape(img)
		flat = img.flatten().reshape(-1,img_dim[0]*img_dim[1]*img_dim[2])
		i = shape_num[shape]
		f.write(bytes([i]))
		#i = color_num[bkgn_color]
		#f.write(bytes([i]))
		#i = color_num[char_color]
		#f.write(bytes([i]))
		i = chars[char]
		f.write(bytes([i]))
		f.write(flat.tobytes())
