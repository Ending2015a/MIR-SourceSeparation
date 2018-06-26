import os
import numpy as np
import tensorflow as tf
import mir_eval
import time

from preprocess import *
from model_2 import CrossNet

#================INPUT ARGUMENTS===========
flags = tf.app.flags

# Path arguments
flags.DEFINE_string('input_list', '../dataset/msp/musdb2_eval_list.txt', 'The list of the data and annotations.')
flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'The directory to save your checkpoints.')
flags.DEFINE_string('checkpoint', '', '')
flags.DEFINE_string('save_path', 'musdb_result.csv', '')

# Training arguments
flags.DEFINE_integer('channels', 5, 'The number of classes to split.')
flags.DEFINE_float('eps', 1e-18, 'epsilon')

# STFT arguments
flags.DEFINE_integer('frame_length', 1024, 'The window length in samples.')
flags.DEFINE_integer('frame_hop', 256, 'The number of samples to step.')
flags.DEFINE_integer('stft_frames', 30, 'The nubmer of frames of the spectrogram.')
flags.DEFINE_integer('sampling_rate', 8192, 'The sampling rate')
flags.DEFINE_integer('overlap', 64, '')

FLAGS = flags.FLAGS

#================PARSE ARGUMENTS============

# Path arguments
input_list = FLAGS.input_list
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

# get input list
data_list, anno_list = read_eval_input_list(input_list)

def main(argv):
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.DEBUG)

        # Get global step
        global_step = tf.get_variable('global_step', [],
                        initializer=tf.constant_initializer(0), trainable=False)

        #============EVALUATING============
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


        st_time = time.time()

        with tf.Session(config=config) as sess:
            
            # create summary writer

            sess.run(tf.global_variables_initializer())

            # Create saver for saving checkpoints of the model
            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)

            if checkpoint != '':
                try:
                    saver.restore(sess, checkpoint)
                    tf.logging.info('Restore checkpoints from {}'.format(checkpoint))
                except:
                    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
                    if not latest_ckpt:
                        tf.logging.info('Restore checkpoints from {}'.format(latest_ckpt))
                    else:
                        saver.restore(sess, latest_ckpt)
                        tf.logging.info('Restore checkpoints from {}'.format(latest_ckpt))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)


            # Create directory if the checkpoint directory does not exist
            checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)


            sdr_list = []
            sir_list = []
            sar_list = []
            perm_list = []

            for idx, (input_file, anno_files) in enumerate(zip(data_list, anno_list)):


                input_data, pad_size, input_samples = get_input_data(input_file,
                                                                     sr=sampling_rate,
                                                                     hop=frame_hop,
                                                                     n_fft=frame_length,
                                                                     frames=stft_frames,
                                                                     overlap=overlap)

                tf.logging.info('[{}/{}] Input data: {} / samples: {} / shape: {} / pad: {}'.format(idx+1, len(input_file),
                                                                        input_file, input_samples, input_data.shape, pad_size))

                channel_list = [list() for n in range(channels)]
                phase_list = []
                for idx, segment in enumerate(input_data):
                    _output, _phase, _norm, _ = inference(sess, segment)
                    phase_list.append( np.squeeze(_phase) )
                    for i in range(channels):
                        channel_list[i].append( np.squeeze(_output[i]) * _norm )

                tf.logging.info('  Reconstructing spectrum...')

                spec = [ reconstruct_spectrum(channel_list[n], phase_list, frames=stft_frames, overlap=overlap, clip=pad_size) for n in range(channels)]
                tf.logging.info('  Reconstructing audio...')

                ys = np.array([ reconstruct_audio(spec[n], hop=frame_hop, fix_length=input_samples) for n in range(channels)])

                anno_sources = np.array([ read(f, sr=8192, fix_length=input_samples) for f in anno_files ])

                (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(anno_sources, ys)
                sdr_list.append(sdr)
                sir_list.append(sir)
                sar_list.append(sar)
                perm_list.append(perm)

                print('sdr: {} / sir: {} / sar: {} / perm: {}'.format(sdr, sir, sar, perm))

            data = {'index':[], 'name':[], 'perm':[]}
            for n in range(channels):
                data['sdr_{:02d}'.format(n)] = []
                data['sir_{:02d}'.format(n)] = []
                data['sar_{:02d}'.format(n)] = []

            for idx, (input_file, sdr, sir, sar, perm) in enumerate(zip(data_list, sdr_list, sir_list, sar_list, perm_list)):
                f = os.path.basename(input_file)
                data['index'].append(idx)
                data['name'].append(f)
                data['perm'].append(' '.join([str(p) for p in perm]))
                for n in range(channels):
                    data['sdr_{:02d}'.format(n)].append(sdr[n])
                    data['sir_{:02d}'.format(n)].append(sir[n])
                    data['sar_{:02d}'.format(n)].append(sar[n])

            import pandas as pd
            df = pd.DataFrame(data=data)
            df.to_csv(save_path, sep='\t')

            tf.logging.info('Complete in {:.6f} sec'.format(time.time()-st_time))

if __name__ == '__main__':
    tf.app.run()

