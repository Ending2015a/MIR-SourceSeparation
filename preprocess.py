import os
import numpy as np
import librosa.core
import tensorflow as tf
import tensorflow.contrib.signal as signal
import re

# read audio
def read(path, sr=22050, mono=True, fix_length=None):
    y, sr = librosa.core.load(path, sr=sr, mono=mono)
    if fix_length:
        if fix_length > len(y):
            y = np.pad(y, (0, fix_length-len(y)), 'constant', constant_values=0)
        else:
            y = y[:fix_length]
    return y

# write wav
def save(path, y, sr=22050):
    if not path.endswith('.wav'):
        path += '.wav'
    librosa.output.write_wav(path, y, sr, norm=True)


# pad and crop audio sources into the given duration
# the unit of the durations are in second (s)
#
# |__________________ audio source __________________|-pad--|
# |--------|
#       |--------|
#             |--------| ....
#                                             .... |--------|
#
#  duration (s)
# |--------|
#  hop_duration (s)
# |-----|

def pad_and_crop(y, sr=22050, duration=4.0, hop_duration=3.0, padding=True):
    
    y_length = y.shape[-1]
    crop_length = int(duration * sr)
    hop_length = int(hop_duration * sr)

    # pad
    pad = crop_length - y_length % hop_length
    pad_shape = [ (0, 0) for n in range(len(y.shape)-1)]
    pad_shape.append((0, pad))
    y_pad = np.pad(y, tuple(pad_shape), 'constant', constant_values=0)

    # crop
    y_list = np.array([ y_pad[..., s:s+crop_length] for s in range(0, y_length, hop_length) ])

    return y_list



def generate_list(datapath, annopath, output):
    data_files = sorted([ l for l in os.listdir(datapath) if l.endswith('all.wav')])
    anno_files = sorted([ l for l in os.listdir(annopath) if l.endswith('.wav')])

    with open(output, 'w') as f:
        for i in range(len(data_files)):
            d = os.path.join(os.path.basename(datapath), data_files[i])
            f.write(d + '\t')
            for j in range(5):
                d = os.path.join(os.path.basename(annopath), anno_files[i*5+j])
                f.write(d + '\t')

            f.write('\n')

# --- tensorflow ops ---

def read_audio(path, sr=8192):
    audio = tf.read_file(path)
    audio = tf.contrib.ffmpeg.decode_audio(audio, file_format='wav', 
                        samples_per_second=sr, channel_count=1)
    audio = tf.squeeze(audio, axis=[-1])
    tf.logging.debug('in read_audio, output audio shape: {}'.format(audio.get_shape().as_list()))
    return audio

# input: a 2D tensor [channels, samples]
# samples: crop length (the input audio samples must larger than this value)
def random_crop_1D_tensors(input, samples=2000):
    shape = input.get_shape().as_list()
    tf.logging.debug('in random_crop, input tensor shape: {}'.format(shape))
    shape[-1] = samples
    output = tf.random_crop(input, shape)
    tf.logging.debug('in random_crop, output tensor shape: {}'.format(output.get_shape()))
    return output

# input: [channels, frames, fft_bins]
def random_crop_2D_tensors(input, frames=256):
    shape = input.get_shape().as_list()
    tf.logging.debug('in random_crop_2D_tensors, input tensor shape: {}'.format(shape))
    shape[-2] = frames
    output = tf.random_crop(input, shape)
    tf.logging.debug('in random_crio_2D_tensors, output tensor shape: {}'.format(output.get_shape()))
    return output

def read_eval_input_list(input_list):
    data_list = []
    anno_list = []
    dir_path = os.path.dirname(input_list)
    with open(input_list, 'r') as f:
        for line in f:
            l = line.split('\t')[:-1]
            data_list.append(os.path.join(dir_path, l[0]))
            anno_list.append([ os.path.join(dir_path, n) for n in l[1:] ])


    return data_list, anno_list

def read_training_input_list(input_list):
    data_list = []
    anno_list = []
    dir_path = os.path.dirname(input_list)
    with open(input_list, 'r') as f:
        for line in f:
            l = line.split('\t')[:-1]
            data_list.append(os.path.join(dir_path, l[0]))
            anno_list.append([ os.path.join(dir_path, n) for n in l[1:] ])

    anno_list = [list(n) for n in zip(*anno_list)]

    tf.logging.debug('in read_input_list, data_list[0]: {}'.format(data_list[0]))

    return data_list, anno_list


def get_samples(hop=256, n_fft=1024, frames=256):
    
    return hop * (frames-1) + n_fft

def get_input_data(filename, sr=8192, n_fft=1024, hop=256, frames=128, overlap=64):
    # read data
    y, sr = librosa.core.load(filename, sr=sr, mono=True)
    y_samples = y.shape[-1]
    y = librosa.util.fix_length(y, y_samples + n_fft//2)
    y = librosa.core.stft(y, n_fft=n_fft, hop_length=hop, window='hann')
    
    _hop = frames - overlap

    # padding
    y_length = y.shape[-1]
    _hop = frames - overlap
    pad = frames - _hop - y_length % _hop
    #pad = frames - y_length % frames
    pad_shape = [ (0,0) for n in range(len(y.shape)-1) ]
    pad_shape.append((0, pad))
    y_pad = np.pad(y, tuple(pad_shape), 'constant', constant_values=0)

    y_length = y_pad.shape[-1]

    tf.logging.info('Input data shape: {} / padded: {}'.format(y.shape, y_pad.shape))

    # crop
    y_list = np.array([ y_pad[..., s:s+frames ] for s in range(0, y_length-overlap, _hop) ])

    return y_list, pad, y_samples


def get_input_tensor(n_fft=1024, hop=256, frames=128):

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[n_fft//2+1, frames])

    resized_tensor = tf.expand_dims(input_tensor, 0)
    resized_tensor = tf.expand_dims(resized_tensor, -1)

    input_magni = tf.abs(resized_tensor)
    input_phase = tf.angle(resized_tensor)

    norm = tf.maximum(tf.sqrt(tf.reduce_max(tf.square(input_magni))), 1e-8)
    input_magni = input_magni/norm
    
    return input_tensor, input_magni, input_phase, norm

def reconstruct_spectrum(input_magni, input_phase, frames=128, overlap=64, clip=0):
    y = None
    multp = None
    hop = frames - overlap

    for idx, (magni, phase) in enumerate(zip(input_magni, input_phase)):

        if y is None:
            shape=( magni.shape[0], hop*len(input_magni)+overlap )
            y = np.zeros(shape=shape, dtype=np.complex128)
            multp = np.zeros(shape=shape, dtype=np.float)

        y[:, idx*hop:idx*hop+frames] += magni * np.exp(1j*phase)
        multp[:, idx*hop:idx*hop+frames] += 1.

    y = y/multp

    assert np.isnan(y).any() == False

    return y[:, :y.shape[1]-clip]


def reconstruct_audio(input_spec, hop=256, fix_length=None):
    y = librosa.core.istft(input_spec, hop_length=hop)
    if fix_length:
        y = librosa.util.fix_length(y, fix_length)
    return y

   


def get_training_input_tensors(data_list, anno_list, hop=256, n_fft=1024, frames=256, batch_size=32):
    
    split_channels = len(anno_list)

    input_slices = tf.train.slice_input_producer(
            [data_list]+anno_list, capacity=batch_size*8, name='slice_input_producer')

    for i in range(len(input_slices)):
        input_slices[i] = read_audio(input_slices[i])

    samples = get_samples(hop, n_fft, frames)

    input_slices = tf.convert_to_tensor(input_slices, dtype=np.float32)
    input_slices = random_crop_1D_tensors(input_slices, samples) # hop * (frames-1) + n_fft
    
    # [channel, frames, n_fft//2+1] -> default [6, 256, 1025]
    input_spec = tf.contrib.signal.stft(input_slices, frame_length=n_fft, 
                                        frame_step=hop, fft_length=n_fft,
                                        window_fn=tf.contrib.signal.hann_window, name='stft')

    tf.logging.debug('in get_training_input_tensors, input_spec shape: {}'.format(input_spec.get_shape()))

    # magnitude spectrogram
    input_magni = tf.abs(input_spec)
    input_phase = tf.angle(input_spec)

    # [channels, frames, freq_bins] -> [channels, freq_bins, frames]
    input_magni = tf.transpose(input_magni, [0, 2, 1])
    input_phase = tf.transpose(input_phase, [0, 2, 1])

    # [channels, freq_bins, frames]
    tf.logging.debug('in get_input_tensors, input_magni shape: {}'.format(input_magni.get_shape()))
    tf.logging.debug('in get_input_tensors, input_phase shape: {}'.format(input_phase.get_shape()))

    # split training data and annotations11
    data_magni = tf.slice(input_magni, [0, 0, 0], [1, -1, -1])
    anno_magni = tf.slice(input_magni, [1, 0, 0], [-1, -1, -1])
    data_phase = tf.slice(input_phase, [0, 0, 0], [1, -1, -1])

    data_magni.set_shape([             1, n_fft//2+1, frames])
    anno_magni.set_shape([split_channels, n_fft//2+1, frames])
    data_phase.set_shape([             1, n_fft//2+1, frames])

    # normalize
    n = tf.maximum(tf.sqrt(tf.reduce_max(tf.square(data_magni))), 1e-8)
    data_magni = data_magni/n
    anno_magni = anno_magni/n

    # tranpose to [freq_bin, time, channels]
    data_magni = tf.transpose(data_magni, [1, 2, 0])
    anno_magni = tf.transpose(anno_magni, [1, 2, 0])
    data_phase = tf.transpose(data_phase, [1, 2, 0])

    # shuffle batch
    data_magni_batch, data_phase_batch, anno_magni_batch = tf.train.shuffle_batch(
            [data_magni, data_phase, anno_magni],
            batch_size = batch_size,
            num_threads = 16,
            capacity = batch_size*8,
            min_after_dequeue=batch_size*5)

    tf.logging.debug('in get_input_tensors, data_magni_batch shape: {}'.format(data_magni_batch.get_shape()))
    tf.logging.debug('in get_input_tensors, data_phase_batch shape: {}'.format(data_phase_batch.get_shape()))
    tf.logging.debug('in get_input_tensors, anno_magni_batch shape: {}'.format(anno_magni_batch.get_shape()))

    # summary
    tf.summary.image('input_magni_spec', data_magni_batch)
    tf.summary.image('input_phase_spec', data_phase_batch)

    return data_magni_batch, data_phase_batch, anno_magni_batch



#tf.logging.set_verbosity(tf.logging.DEBUG)

#data_list, anno_list = read_training_input_list('../dataset/msp/song001/input_list.txt')
#get_training_input_tensors(data_list, anno_list)


