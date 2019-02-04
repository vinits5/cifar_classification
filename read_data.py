import cPickle
import cv2
import numpy as np
import sys

with open('cifar-10-batches-py/test_batch','rb') as fo:
	data = cPickle.load(fo)

with open('cifar-10-batches-py/batches.meta','rb') as fo:
	label_names = cPickle.load(fo)

idx = int(sys.argv[1])

def display_image(idx):
	img = data['data'][idx]
	img = np.array(img)

	img_reshape = np.zeros((32,32,3),dtype=np.int8)
	array_size = 1024
	img_size = 32

	img_r = img[0:1024].reshape((32,32))
	img_g = img[1024:2*1024].reshape((32,32))
	img_b = img[2*1024:3*1024].reshape((32,32))

	img = np.dstack((img_r,img_g,img_b))

	label_idx = data['labels'][idx]
	print(label_idx)
	print('Class of Image: '+label_names['label_names'][label_idx])

	cv2.imshow('Image',img)
	cv2.waitKey(10000)

if __name__=='__main__':
	display_image(idx)