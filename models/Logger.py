import tensorflow as tf

class Logger:
	def __init__(self, path, sess):
		self.writer = tf.summary.FileWriter(path, sess.graph)

	def log_scalar(self, tag='Test', value=0, step=0):
		summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
		self.writer.add_summary(summary, step)