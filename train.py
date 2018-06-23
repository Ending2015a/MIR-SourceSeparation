import os
import numpy as np
import tensorflow as tf

from preprocess import *
from model_2 import CrossNet

#================INPUT ARGUMENTS===========
flags = tf.app.flags

# Path arguments
flags.DEFINE_string('input_list', '../dataset/msp/input_list.txt', 'The list of the data and annotations.')
flags.DEFINE_string('log_dir', './log/', 'The log directory to save your summary and event files.')
flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'The directory to save your checkpoints.')

# Training arguments
flags.DEFINE_integer('channels', 5, 'The number of classes to split.')
flags.DEFINE_integer('batch_size', 32, 'The batch size for training.')
flags.DEFINE_integer('eval_batch_size', 32, 'The batch size used for validation.')
flags.DEFINE_integer('num_epochs', 1000, 'The number of epochs to train your model.')
flags.DEFINE_integer('num_epochs_before_decay', 100, 'The number of epochs before decaying your learning rate.')
flags.DEFINE_float('weight_decay', 2e-4, 'The weight decay for conv layers.')
flags.DEFINE_float('learning_rate_decay_factor', 0.1, 'The learning rate decay factor.')
flags.DEFINE_float('initial_learning_rate', 1e-5, 'The initial learning rate for your training.')
flags.DEFINE_float('minimum_learning_rate', 1e-10, 'The minimum learning rate')
flags.DEFINE_string('optimizer', 'adadelta', 'The optimizer to train your model (adam/adadelta/rmsprop)')
flags.DEFINE_string('cost_function', 'se', 'The cost function to evaluate the loss of your model (se/mse/bce/l1)')
flags.DEFINE_float('eps', 1e-18, 'epsilon')
flags.DEFINE_boolean('alpha_loss', False, 'Enable alpha loss (not implemented yet)')
flags.DEFINE_boolean('beta_loss', False, 'Enable beta loss (not implemented yet)')
flags.DEFINE_float('alpha_factor', 1e-4, 'The factor of the alpha loss (not implemented yet)')
flags.DEFINE_float('beta_factor', 1e-4, 'The factor of the beta loss (not implemented yet)')

# Summary arguments
flags.DEFINE_integer('summary_step', 100, 'The number of steps to write the summary')

# STFT arguments
flags.DEFINE_integer('frame_length', 1024, 'The window length in samples.')
flags.DEFINE_integer('frame_hop', 256, 'The number of samples to step.')
flags.DEFINE_integer('stft_frames', 30, 'The nubmer of frames of the spectrogram.')

FLAGS = flags.FLAGS

#================PARSE ARGUMENTS============

# Path arguments
input_list = FLAGS.input_list
log_dir = FLAGS.log_dir
checkpoint_dir = FLAGS.checkpoint_dir

# Training arguments
channels = FLAGS.channels
batch_size = FLAGS.batch_size
eval_batch_size = FLAGS.eval_batch_size
num_epochs = FLAGS.num_epochs
num_epochs_before_decay = FLAGS.num_epochs_before_decay
weight_decay = FLAGS.weight_decay
learning_rate_decay_factor = FLAGS.learning_rate_decay_factor
initial_learning_rate = FLAGS.initial_learning_rate
minimum_learning_rate = FLAGS.minimum_learning_rate
optimizer = FLAGS.optimizer
cost_function = FLAGS.cost_function
eps = FLAGS.eps

# Summary arguments
summary_step = FLAGS.summary_step

# STFT arguments
frame_length = FLAGS.frame_length
frame_hop = FLAGS.frame_hop
stft_frames = FLAGS.stft_frames
#==========PREPARATION FOR TRAINING=========

# get input list
data_list, anno_list = read_training_input_list(input_list)

# compute 
num_batches_per_epoch = int(len(data_list) / batch_size) + 1
num_steps_per_epoch = num_batches_per_epoch
decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

grad_and_var = {}

def split_tensor_channels(tensor):
    slice_shape = tensor.get_shape().as_list()
    channels = slice_shape[-1]
    slice_shape[-1] = 1

    zero_list = [0] * (len(slice_shape)-1)
    channel_list = []
    for i in range(channels):
        cn = tf.slice(tensor, zero_list + [i], slice_shape)
        channel_list.append(cn)

    return channel_list

def square_loss(logits, label, alpha=1.):
    err_ = alpha * tf.reduce_mean( tf.reduce_sum(tf.square(logits-label, 'square_error'), axis=[1, 2, 3]) )
    tf.add_to_collection(tf.GraphKeys.LOSSES, err_)
    return err_

def mean_square_loss(logits, label, alpha=1.):
    err_ = alpha * tf.losses.mean_squared_error(label, logits, 
                                                weights=alpha,
                                                loss_collection=tf.GraphKeys.LOSSES)
    return err_


def bce_loss(logits, label, alpha=1.):
    logits = tf.nn.sigmoid(logits)
    err_ = alpha * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits))
    tf.add_to_collection(tf.GraphKeys.LOSSES, err_)
    return err_

def l1_norm(logits, label, alpha=1.):
    err_ = alpha * tf.reduce_sum(tf.abs(tf.subtract(logits, label)))
    tf.add_to_collection(tf.GraphKeys.LOSSES, err_)
    return err_

def create_loss(input_label, magni_logits, magni_label):

    cost_fun = square_loss #se
    if cost_function == 'mse':
        cost_fun = mean_square_loss
    elif cost_function == 'bce':
        cost_fun = bce_loss
    elif cost_function == 'l1':
        cost_fun = l1_norm

    for i in range(channels):
         # compute loss
        recon_loss = cost_fun(magni_logits[i], magni_label[i])
        mse = tf.losses.mean_squared_error(magni_logits[i], magni_label[i], loss_collection='mse')

        # logging
        tf.logging.debug('in create_loss, magni_logits[{}] shape: {}'.format(i, magni_logits[i].get_shape()))
        tf.logging.debug('in create_loss, magni_label[{}] shape: {}'.format(i, magni_label[i].get_shape()))

        # write summary
        tf.summary.image('output_magni_spec_c{}'.format(i), magni_logits[i], max_outputs=1)
        tf.summary.image('anno_magni_spec_c{}'.format(i), magni_label[i], max_outputs=1)
        tf.summary.scalar('loss_{}'.format(i), recon_loss)
        tf.summary.scalar('mse_{}'.format(i), mse)

    # sum
    total_magni = tf.reduce_sum(tf.concat(magni_logits, axis=-1), axis=-1, keepdims=True)
    recon_sum_loss = cost_fun(total_magni, input_label)

    # write summary
    tf.summary.scalar('loss_sum', recon_sum_loss)
    tf.summary.image('output_magni_spec_sum', total_magni, max_outputs=1)

    # total mse
    mses = tf.add_n(tf.get_collection('mse'), name='total_mse')
    tf.summary.scalar('total_mse', mses)
    # weight decay
    #for var in tf.get_collection('WEIGHT_DECAY_VARIABLE'):
    #    tf.add_to_collection(tf.GraphKeys.LOSSES, weight_decay * tf.nn.l2_loss(var))

    # total loss
    loss_ = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES), name='total_loss')

    # add to summary
    tf.summary.scalar('total_loss', loss_)

    return loss_

def create_learning_rate(global_step):

    lr = tf.train.exponential_decay(
                    learning_rate = initial_learning_rate,
                    global_step = global_step,
                    decay_steps = decay_steps,
                    decay_rate = learning_rate_decay_factor)

    lr = tf.maximum(lr, minimum_learning_rate)
    
    # add to summary
    tf.summary.scalar('learning_rate', lr)

    return lr

def create_train_op(loss, learning_rate, global_step):
    if optimizer == 'adam':
        _optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=eps)
    elif optimizer == 'adadelta':
        _optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, epsilon=eps)
    elif optimizer == 'rmsprop':
        _optimizer = tf.train.PMSPropOptimizer(learning_rate=learning_rate, epsilon=eps)

    grads = _optimizer.compute_gradients(loss)
    for grad, var in grads:
        grad_and_var[var.op.name] = grad
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    train_op = _optimizer.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    return train_op

def create_summary_op():
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    summary_op = tf.summary.merge_all()

    return summary_op

def test_nan_and_zero(y, name):
    print('{} contains any nan? : {}'.format(name, np.isnan(np.array(y)).any()) )
    print('{} contains any zero? : {}'.format(name, (np.array(y)==0).any()))

def main(argv):
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.DEBUG)

        # Get global step
        global_step = tf.get_variable('global_step', [],
                        initializer=tf.constant_initializer(0), trainable=False)

        #============TRAINING============
        # Preparing tranining data
        with tf.name_scope('preprocess') as scope:
            data_magni_batch, data_phase_batch, anno_magni_batch = get_training_input_tensors(data_list, 
                                                                                    anno_list, hop=frame_hop, 
                                                                                    n_fft=frame_length, 
                                                                                    frames=stft_frames, 
                                                                                    batch_size=batch_size)


        # Construct network
        with tf.variable_scope('crossnet', reuse=tf.AUTO_REUSE):
            model = CrossNet(inputs={'magni': data_magni_batch}, 
                             channels=channels,
                             eps=eps)
            magni_masks = model.build_model()
            magni_logits = model.build_post_model()


        split_anno_magni_batch = split_tensor_channels(anno_magni_batch)

        # create total loss
        loss = create_loss(data_magni_batch, magni_logits, split_anno_magni_batch)
        # create learning rate
        lr = create_learning_rate(global_step)

        # create training operation
        train_op = create_train_op(loss, lr, global_step)
        # create summary operation
        summary_op = create_summary_op()

        # tranining step
        def train_step(sess, train_op, global_step):
            
            import time

            start_time = time.time()
            lists = []

            _, _loss, _global_step, _lr, = sess.run([train_op, loss, global_step, lr], feed_dict={model.use_dropout: 1.})
            time_elapsed = time.time() - start_time

            #test_nan_and_zero(_dbg, 'conv1')
            #test_nan_and_zero(_dbg2, 'conv1_bn')
            #test_nan_and_zero(_dbg3, 'reduce_m')

            return _loss, int(_global_step), _lr, time_elapsed

        #============VALIDATION==========
        # TODO:
        # with tf.variable_scope('crossnet', reuse=tf.AUTO_REUSE):
        # 

        # limit gpu memory usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            
            # create summary writer
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

            sess.run(tf.global_variables_initializer())

            # Create saver for saving checkpoints of the model
            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)

            # Trying to restore variables
            latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
            if not latest_ckpt:
                print('Checkpoint not found in {}'.format(checkpoint_dir))
            else:
                # restore
                saver.restore(sess, latest_ckpt)
                print('Restore checkpoints from {}'.format(latest_ckpt))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            print('start training...')

            import time

            # Create directory if the checkpoint directory does not exist
            checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            # training...
            try:
                for epoch in range(num_epochs):
                    tf.logging.info('Epoch {}/{}'.format(epoch+1, num_epochs))
                    for step in range(num_steps_per_epoch):
                        _loss, _global_step, _lr, _time = train_step(sess, train_op, global_step)
                        tf.logging.info('[Epoch {0}/{1} / Step {2}/{3} / Global Step {4}] loss {5:.4f} / learning rate {6:E} / batch {7} / {8:.2f} sec/step'.format(
                                        epoch+1, num_epochs, step+1, num_steps_per_epoch, int(_global_step), _loss, _lr, batch_size, _time))
                    
                        # Write to summary
                        if _global_step % summary_step == 0:
                            summary = sess.run(summary_op)
                            summary_writer.add_summary(summary, _global_step)
                            tf.logging.info('write to summary')

                    save_path = saver.save(sess, checkpoint_path, global_step=global_step)
                    tf.logging.info('The model has been saved to: {}'.format(save_path))

            except KeyboardInterrupt:
                sel = input('Do you want to save the model? [y/n]: ')
                if sel == 'y':
                    save_path = saver.save(sess, checkpoint_path, global_step=global_step)
                    tf.logging.info('The model has been saved to: {}'.format(save_path))


if __name__ == '__main__':
    tf.app.run()

