import math
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import DataProcessor
import Utils

LEARNING_RATE = 0.01
SKIP_STEP = 500
LOG_DIR = './graphs'
DATA_BY_LR = os.path.join(LOG_DIR, 'improved_graph/lr' + str(LEARNING_RATE))
METADATA = os.path.join(LOG_DIR, 'metadata.tsv')
MODEL_CKPT = os.path.join(LOG_DIR, "model.ckpt")
# DATA_FILENAME = 'sample_files/UKDALE-7month-2.0mins-building1_start_stop.csv'
#83500
#  DATA_FILENAME = 'sample_files/uk_dale/UKDALE-31-3-2013-31-12-2013-1.0mins-building1_start_stop.csv'
# DATA_FILENAME = 'sample_files/uk_dale/UKDALE-31-12-2013-31-12-2014-1.0mins-building1_start_stop.csv'
# DATA_FILENAME = 'sample_files/uk_dale/UKDALE-31-12-2014-31-12-2015-1.0mins-building1_start_stop.csv'
# 51499
# DATA_FILENAME = 'sample_files/uk_dale/UKDALE-31-12-2015-31-12-2016-1.0mins-building1_start_stop.csv'
# DATA_FILENAME = 'sample_files/4month-2mins_start_stop.csv'
# DATA_FILENAME = 'sample_files/1month-2mins-startstop.csv'
#DATA_FILENAME = '../tokenization/labels.csv'
DATA_FILENAME = "../tokenization/energy_tokens_sequence.csv"

BATCH_SIZE = 64
EMBEDDING_SIZE = 300
SKIP_WINDOW = 6
TRAINING_STEPS = 83999
# TRAINING_STEPS = 51499
NUM_SAMPLED = 5


class SkipGramModel:
    def __init__(self, vocabulary_size, embedding_size, batch_size, num_sampled, learning_rate, valid_window=52,
                 valid_size=10):
        self.vocabulary_size = vocabulary_size
        self.learning_rate = learning_rate
        self.num_sampled = num_sampled
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.valid_size = valid_size
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    def _define_placeholders(self):
        with tf.name_scope('inputs'):
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size], name='X')
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='labels')
            self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32, name='validation')

    def _define_embedding(self):
        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            with tf.name_scope('lookupembeddings'):
                self.embeddings = tf.Variable(
                    tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0), name='embeddings')

    def _define_loss(self):
        with tf.device('/cpu:0'):
            with tf.name_scope('loss'):
                embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
                nce_weights = tf.Variable(
                    tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                        stddev=1.0 / math.sqrt(self.embedding_size)))
                nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
                self.loss = tf.reduce_mean(
                    tf.nn.nce_loss(weights=nce_weights,
                                   biases=nce_biases,
                                   labels=self.train_labels,
                                   inputs=embed,
                                   num_sampled=self.num_sampled,
                                   num_classes=self.vocabulary_size), name='loss')

    def _define_optimizer(self):
        with tf.device('/cpu:0'):
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss,
                                                                                    global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._define_placeholders()
        self._define_embedding()
        self._define_loss()
        self._define_optimizer()
        self._create_summaries()


def train_model(model, batch_gen, num_steps):
    saver = tf.train.Saver()
    initial_step = 0
    Utils.make_dir('checkpoints')
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
        total_loss = 0.0
        writer = tf.summary.FileWriter(LOG_DIR, session.graph)
        initial_step = model.global_step.eval()
        for step in range(initial_step, initial_step + num_steps):
            batch_inputs, batch_labels = next(batch_gen)
            feed_dict = {model.train_inputs: batch_inputs, model.train_labels: batch_labels}
            summary, _, loss_batch = session.run([model.summary_op, model.optimizer, model.loss],
                                                 feed_dict=feed_dict)
            writer.add_summary(summary, step)
            total_loss += loss_batch
            if (step + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(step, total_loss / SKIP_STEP))
                total_loss = 0.0
                saver.save(session, 'checkpoints/skip-gram', step)
        final_embed_matrix = session.run(model.embeddings)
        embedding_var = tf.Variable(final_embed_matrix[:1000], name='embedding')
        session.run(embedding_var.initializer)
        config = projector.ProjectorConfig()
        # summary_writer = tf.summary.FileWriter(LOG_DIR)
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(writer, config)
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(session, MODEL_CKPT, 1)


def main():
    batch_gen = DataProcessor.process_gmm_data(DATA_FILENAME, BATCH_SIZE, SKIP_WINDOW)
    energy_states_size = DataProcessor.get_number_of_energy_states(DATA_FILENAME)
    vocabulary = DataProcessor.build_vocab(np.arange(energy_states_size))
    # device_states_size = get_device_states_size(DATA_FILENAME)
    model = SkipGramModel(energy_states_size, EMBEDDING_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
    model.build_graph()
    train_model(model, batch_gen, TRAINING_STEPS)


if __name__ == '__main__':
    main()
