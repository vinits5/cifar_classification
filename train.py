import tensorflow as tf 
import numpy as np 
import os
import sys
import helper
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
import argparse
import importlib


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='no_mode', help='mode: train or test')
parser.add_argument('--model', type=str, default='classifier_VGG', help='model for training')
parser.add_argument('--log_dir', type=str, default='log_trial5', help='name of log directory')
parser.add_argument('--img_size', type=int, default=32, help='size of image')
parser.add_argument('--channels', type=int, default=3, help='channels in image')
parser.add_argument('--learning_rate', type=float, default=1e-04, help='initial learning rate')
parser.add_argument('--decay_rate', type=float, default=0.7, help='decay rate for lr decay')
parser.add_argument('--decay_steps', type=float, default=500, help='decay steps for lr decay')
parser.add_argument('--max_epoch', type=int, default=20, help='maximum number of epochs for training')
parser.add_argument('--batch_size', type=int, default=100, help='batch size for training')
parser.add_argument('--model_path', type=str, default='', help='weights for testing')

FLAGS = parser.parse_args()

MODEL = importlib.import_module(FLAGS.model)
Logger = importlib.import_module('Logger')
# Params
IMG_SIZE = FLAGS.img_size
CHANNELS = FLAGS.channels
Init_LEARNING_RATE = FLAGS.learning_rate
DECAY_RATE = FLAGS.decay_rate
DECAY_STEP = FLAGS.decay_steps
MAX_EPOCH = FLAGS.max_epoch

if FLAGS.mode == 'train':
	BATCH_SIZE = FLAGS.batch_size
else:
	BATCH_SIZE = 1

LOG_DIR = FLAGS.log_dir

if FLAGS.mode == 'train':
	if not os.path.exists(LOG_DIR):
		os.mkdir(LOG_DIR)
		os.mkdir(os.path.join(LOG_DIR,'train'))
		os.mkdir(os.path.join(LOG_DIR,'test'))
	os.system('cp train.py %s'%(LOG_DIR))
	os.system('cp helper.py %s'%(LOG_DIR))
	os.system('cp -a models/ %s/'%(LOG_DIR))

def get_learning_rate(step):
	learning_rate = tf.train.exponential_decay(learning_rate=Init_LEARNING_RATE,
												global_step=step,
												decay_steps=DECAY_STEP,
												decay_rate=DECAY_RATE,
												staircase=True)
	learning_rate = tf.maximum(learning_rate, 1e-05) # CLIP THE LEARNING RATE!
	return learning_rate

def log_string(info):
	file = open(os.path.join(LOG_DIR,'log_text.txt'),'a')
	file.write(info)
	file.write('\n')
	file.close()
	print(info)

def train():
	with tf.device('/device:GPU:0'):
		is_training = tf.placeholder(tf.bool, shape=())
		ip_img, gt_prediction = MODEL.get_placeholders(BATCH_SIZE, IMG_SIZE)
		prediction = MODEL.get_model(ip_img, is_training)
		loss = MODEL.get_loss(prediction, gt_prediction)

		with tf.variable_scope('Training_Params'):
			step = tf.Variable(0)
			learning_rate = get_learning_rate(step)

		with tf.variable_scope('Optimizer'):
			trainer = tf.train.AdamOptimizer(learning_rate)
			train_op = trainer.minimize(loss, global_step=step)
		saver = tf.train.Saver()

	ops = {'input':ip_img,
			'prediction':prediction,
			'gt_prediction':gt_prediction,
			'loss':loss,
			'step':step,
			'learning_rate':learning_rate,
			'train_op':train_op,
			'is_training':is_training}

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	config.log_device_placement = False
	sess = tf.Session(config=config)

	init = tf.global_variables_initializer()
	sess.run(init)

	logger_train = Logger.Logger(os.path.join(LOG_DIR,'train'), sess)
	logger_test = Logger.Logger(os.path.join(LOG_DIR,'test'), sess)

	if FLAGS.mode == 'train':
		log_string(str(FLAGS)+'\n')
		for epoch in range(MAX_EPOCH):
			log_string("#####%3d#####"%epoch)
			train_one_epoch(sess, ops, logger_train)
			log_string(' ')
			eval_one_epoch(sess, ops, logger_test)
			log_string(' ')
			saver.save(sess,os.path.join(LOG_DIR,'model.ckpt'))
			if epoch%4==0:
				saver.save(sess,os.path.join(LOG_DIR,'model'+str(epoch)+'.ckpt'))

	if FLAGS.mode == 'test':
		test_network(sess, ops, saver, FLAGS.model_path)

def train_one_epoch(sess, ops, logger_train):
	is_training = True
	files = helper.data_files()
	file_idxs = np.arange(0,len(files))
	np.random.shuffle(file_idxs)
	# file_idxs = [0]

	for fn in file_idxs:
		data, labels = helper.read_files(files[fn])
		data, labels = helper.shuffle_data(data, labels)
		# data = data[0:10000,:]
		# labels = labels[0:10000]
		num_batches = data.shape[0]//BATCH_SIZE
		total_seen = 0
		total_correct = 0
		total_loss = 0
		log_string('----' + str(fn) + '-----')

		for idx in range(num_batches):
			start_idx = idx*BATCH_SIZE
			end_idx = (idx+1)*BATCH_SIZE

			current_data = helper.preprocess_data(data[start_idx:end_idx])
			current_labels = np.array(labels[start_idx:end_idx]).reshape((BATCH_SIZE))

			feed_dict = {ops['input']: current_data,
						 ops['gt_prediction']: current_labels,
						 ops['is_training']: is_training}

			pred_val, loss_val, step, learning_rate, _ = sess.run([ops['prediction'], 
								ops['loss'], ops['step'], ops['learning_rate'], ops['train_op']], feed_dict=feed_dict)

			logger_train.log_scalar(tag='Loss',value=loss_val,step=step)

			pred_val = np.argmax(pred_val, 1)
			correct = np.sum(pred_val == current_labels)
			total_correct += correct
			total_seen += BATCH_SIZE
			total_loss += loss_val
			print("Batch: {}, Correct: {} & Loss: {}\r".format(idx, correct, loss_val)),
			sys.stdout.flush()
		logger_train.log_scalar(tag='Total Loss',value=total_loss/float(num_batches),step=step)
		logger_train.log_scalar(tag='Learning Rate',value=learning_rate,step=step)
		logger_train.log_scalar(tag='Accuracy',value=total_correct/float(total_seen),step=step)
		log_string('Total Training Loss: {}'.format(total_loss/float(num_batches)))
		log_string('Training Accuracy: {}'.format(total_correct/float(total_seen)))

def eval_one_epoch(sess, ops, logger_test):
	is_training = False
	log_string('###### TEST ######')

	data, labels = helper.read_files('test_batch')
	# data = data[0:100,:]
	# labels = labels[0:100]
	num_batches = data.shape[0]//BATCH_SIZE
	total_seen = 0
	total_correct = 0
	total_loss = 0

	for idx in range(num_batches):
		start_idx = idx*BATCH_SIZE
		end_idx = (idx+1)*BATCH_SIZE

		current_data = helper.preprocess_data(data[start_idx:end_idx])
		current_labels = np.array(labels[start_idx:end_idx]).reshape((BATCH_SIZE))

		feed_dict = {ops['input']: current_data,
					 ops['gt_prediction']: current_labels,
					 ops['is_training']: is_training}

		pred_val, loss_val, step, learning_rate = sess.run([ops['prediction'], 
							ops['loss'], ops['step'], ops['learning_rate']], feed_dict=feed_dict)

		logger_test.log_scalar(tag='Loss',value=loss_val,step=step)

		pred_val = np.argmax(pred_val, 1)
		correct = np.sum(pred_val == current_labels)
		total_correct += correct
		total_seen += BATCH_SIZE
		total_loss += loss_val
		print("Batch: {}, Correct: {} & Loss: {}\r".format(idx, correct, loss_val)),
		sys.stdout.flush()
	logger_test.log_scalar(tag='Total Loss',value=total_loss/float(num_batches),step=step)
	logger_test.log_scalar(tag='Accuracy',value=total_correct/float(total_seen),step=step)
	log_string('Total Test Loss: {}'.format(total_loss/float(num_batches)))
	log_string('Test Accuracy: {}'.format(total_correct/float(total_seen)))

def test_network(sess, ops, saver, model_path):
	saver.restore(sess, model_path)

if __name__=='__main__':
	if FLAGS.mode == 'no_mode':
		print('Specify mode: train or test')
	else:
		train()