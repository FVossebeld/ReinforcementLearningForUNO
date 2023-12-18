import tensorflow as tf
from datetime import datetime


class TensorflowLogger:
    def __init__(self, log_dir):
        # Initialize the logger
        self.timestamp = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
        self.log_dir = log_dir
        self.variable_steps = {}

    def scalar(self, name, value, step=None):
        # Get the step value for the current scalar
        if step is None:
            if name in self.variable_steps:
                step = self.variable_steps[name] = self.variable_steps[name] + 1
            else:
                step = self.variable_steps[name] = 0
        else:
            self.variable_steps[name] = step

        # Create the summary and write it
        with tf.summary.create_file_writer(self.log_dir + '/summary_' + self.timestamp).as_default():
            tf.summary.scalar(name, value, step=step)
            tf.summary.flush()

    def flush(self):
        # Flush the file writer
        with tf.summary.create_file_writer(self.log_dir + '/summary_' + self.timestamp).as_default():
            tf.summary.flush()
