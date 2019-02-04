import tensorflow as tf
import numpy as np

# Used to declare placeholders for input images and labels.
def get_placeholders(BATCH_SIZE, IMG_SIZE):
	# Arguments:
		# BATCH_SIZE:		size of batch
		# IMG_SIZE:			width/height of image (assumption: width=height)
	# Output:
		# ip_img:			Placeholder for image (batch_size x width x height x 3)
		# gt_prediction:	Placeholder for labels	(batch_size)
	with tf.variable_scope('Inputs'):
		ip_img = tf.placeholder(tf.float32,shape=(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3))
		gt_prediction = tf.placeholder(tf.int32,shape=(BATCH_SIZE))
	return ip_img, gt_prediction

# Used to define convolution layers.
def convolution(ip_img, layers):
	# Arguments:
		# ip_img:		Input for first layer. (batch_size x width x height x 3)
		# layers:		Dictionary of layer information {idx: [num_outputs, kernel_size, stride]}
	# Output:
		# net:			Output of convolution layers.
	net = ip_img
	for idx in range(1, len(layers)+1):
		net = tf.contrib.layers.conv2d(inputs=net, num_outputs=layers[idx][0], kernel_size=layers[idx][1], stride=layers[idx][2], padding="SAME", scope='conv_'+str(idx))
	return net

# Used to define fully connected layers.
def fcn(flat, layers, is_training):
	# Arguments:
		# flat:			Input for first layer.
		# layers:		List of layer information [num_outputs]
	# Output:
		# net:			Output of fully connected layers.
	net = flat
	for idx in range(len(layers)):
		net = tf.contrib.layers.fully_connected(inputs=net, num_outputs=layers[idx], activation_fn=tf.nn.relu, scope='fcn_'+str(idx+1))
		net = tf.contrib.layers.dropout(inputs=net, keep_prob=0.5, is_training=is_training, scope='dp_'+str(idx+1))
	return net

# Used to define network structure.
def get_model(ip_img, is_training):
	# Argument:
		# ip_img:		Tensor having input image (batch_size x width x height x 3)
	# Output:
		# prediction:	Prediction probabilities for each class (batch_size x num_classes)

	# Ref. VGG Network
	with tf.variable_scope('CNN') as _:
		layers = {1: [64, 3, 1],
				  2: [64, 3, 2],
				  3: [128, 3, 1],
				  4: [128, 3, 2],
				  5: [256, 3, 1],
				  6: [256, 3, 1],
				  7: [256, 3, 2],
				  8: [512, 3, 1],
				  9: [512, 3, 1],
				  10: [512, 3, 2],
				  11: [512, 3, 1],
				  12: [512, 3, 1],
				  13: [512, 3, 2]}
		net = convolution(ip_img, layers)
		flat = tf.contrib.layers.flatten(net, scope='flatten')
	with tf.variable_scope('FCN') as _:
		layers = [4096, 4096]
		net = fcn(flat, layers, is_training)
		prediction = tf.contrib.layers.fully_connected(inputs=net, num_outputs=10, activation_fn=None, scope='fcn_'+str(len(layers)+1))
	return prediction

# Used to calculate loss.
def get_loss(prediction, gt_prediction):
	# Arguments:
		# prediction:		Prediction probabilities for each class (batch_size x num_classes)
		# gt_prediction:	Actual labels for each image in batch (batch_size)
	# Output:
		# loss:				Softmax cross entropy.
	with tf.variable_scope('Loss'):
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt_prediction)
		loss = tf.reduce_mean(loss)
	return loss