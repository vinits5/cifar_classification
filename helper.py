import numpy as np 
import os
import cPickle

def data_files():
	files = []
	for i in range(1,6):
		files.append('data_batch_'+str(i))
	return files

def read_files(file_name):
	with open(os.path.join('cifar-10-batches-py',file_name),'rb') as fo:
		data = cPickle.load(fo)
	return data['data'],data['labels']

def shuffle_data(data,labels):
	idxs = np.arange(0,len(labels))
	np.random.shuffle(idxs)
	shuffled_data = np.zeros(data.shape)
	shuffled_labels = []
	for i in range(len(labels)):
		shuffled_data[i,:]=data[idxs[i],:]
		shuffled_labels.append(labels[idxs[i]])
	return shuffled_data,shuffled_labels

def preprocess_data(images):
	processed_images = np.zeros((images.shape[0],32,32,3))
	for i in range(images.shape[0]):
		img = images[i]
		img = np.array(img)

		img_r = img[0:1024].reshape((32,32))
		img_g = img[1024:2*1024].reshape((32,32))
		img_b = img[2*1024:3*1024].reshape((32,32))

		img = np.dstack((img_r,img_g,img_b))
		processed_images[i,:,:,:] = img
	return processed_images
