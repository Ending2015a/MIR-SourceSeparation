import os
import numpy as np
import tensorflow as tf

import librosa.core
from model_2 import CrossNet
from preprocess import *

#================INPUT ARGUMENTS===========
flags = tf.app.flags

# Path arguments
flags.DEFINE_string('input_file', '', 'The input file.')
flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'The directory to save your checkpoints.')
flags.DEFINE_string('checkpoint', '', '')
flags.DEFINE_string('save_path', '', 'The save path.')
# Training arguments
flags.DEFINE_integer('channels', 5, 'The number of classes to split.')
flags.DEFINE_float('eps', 1e-18, 'epsilon')

# STFT arguments
flags.DEFINE_integer('frame_length', 1024, 'The window length in samples.')
flags.DEFINE_integer('frame_hop', 256, 'The number of samples to step.')
flags.DEFINE_integer('stft_frames', 30, 'The nubmer of frames of the spectrogram.')
flags.DEFINE_integer('sampling_rate', 8192, 'The sampling rate.')
flags.DEFINE_integer('overlap', 64, '')

FLAGS = flags.FLAGS

#================PARSE ARGUMENTS============

# Path arguments
input_file = FLAGS.input_file
checkpoint_dir = FLAGS.checkpoint_dir
checkpoint = FLAGS.checkpoint
save_path = FLAGS.save_path

# Training arguments
channels = FLAGS.channels
eps = FLAGS.eps

# STFT arguments
frame_length = FLAGS.frame_length
frame_hop = FLAGS.frame_hop
stft_frames = FLAGS.stft_frames
sampling_rate = FLAGS.sampling_rate
overlap = FLAGS.overlap

#==========PREPARATION FOR TRAINING=========


def main(argv):
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.DEBUG)

        # Get global step
        global_step = tf.get_variable('global_step', [],
                        initializer=tf.constant_initializer(0), trainable=False)

        #============INFERENCE============
        # Preparing tranining data
        input_tensor, data_magni, data_phase, data_norm = get_input_tensor(n_fft=frame_length,
                                                             hop=frame_hop,
                                                             frames=stft_frames)

        # Construct network
        with tf.variable_scope('crossnet', reuse=tf.AUTO_REUSE):
            model = CrossNet(inputs={'magni': data_magni}, 
                             channels=channels,
                             eps=eps)
            magni_masks = model.build_model()
            magni_logits = model.build_post_model()
        


        def inference(sess, input_data):
            
            import time
            start_time = time.time()

            _output = sess.run([data_phase, data_norm] + magni_logits, feed_dict={input_tensor: input_data})
            time_elapsed = time.time() - start_time

            _phase = _output[0]
            _norm = _output[1]
            _output = _output[2:]

            return _output, _phase, _norm, time_elapsed



        # limit gpu memory usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(tf.global_variables_initializer())

            # Create saver for saving checkpoints of the model
            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)

            if checkpoint != '':
                try:
                    saver.restore(sess, checkpoint)
                    tf.logging.info('Restore checkpoints from {}'.format(checkpoint))
                except:
                    # Trying to restore variables
                    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
                    if not latest_ckpt:
                        tf.logging.info('Checkpoint not found in {}'.format(checkpoint_dir))
                    else:
                        # restore
                        saver.restore(sess, latest_ckpt)
                        tf.logging.info('Restore checkpoints from {}'.format(latest_ckpt))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            import time

            # Create directory if the checkpoint directory does not exist
            checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            tf.logging.info('Generating input data...')

            # prepare data
            input_data, pad_size = get_input_data(input_file,
                                                  sr=sampling_rate,
                                                  hop=frame_hop, 
                                                  n_fft=frame_length,
                                                  frames=stft_frames,
                                                  overlap=overlap)

            tf.logging.info('Input data: {} / shape: {} / pad: {}'.format(input_file, input_data.shape, pad_size))

            # inference...
            channel_list = [list() for n in range(channels)]
            phase_list = []
            for idx, segment in enumerate(input_data):
                tf.logging.info('Seg {}/{}'.format(idx+1, len(input_data)))
                
                _output, _phase, _norm, _time = inference(sess, segment)
                tf.logging.info('[Seg {0}/{1}] {2:.2f} sec/seg'.format(idx+1, len(input_data), _time))

                
                phase_list.append( np.squeeze(_phase) )
                # append to each channel
                for i in range(channels):
                    channel_list[i].append( np.squeeze(_output[i]) * _norm)

            tf.logging.info('Reconstructing spectrum...')

            spec = [ reconstruct_spectrum(channel_list[n], phase_list, frames=stft_frames, overlap=overlap, clip=pad_size) for n in range(channels) ]
            tf.logging.info('spectrum shape: {}'.format(np.array(spec).shape))

            tf.logging.info('Reconstructing audio...')

            ys = [ reconstruct_audio(spec[n], hop=frame_hop) for n in range(channels)]

            for n in range(channels):
                path, _ = os.path.splitext(os.path.basename(input_file))
                path = save_path + path + '_{}.wav'.format(n+1)
                tf.logging.info('Saving channel {} to: {}'.format(n+1, path))
                save(path, ys[n], sr=sampling_rate)


if __name__ == '__main__':
    tf.app.run()

