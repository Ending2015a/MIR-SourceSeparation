import os
import numpy as np
import tensorflow as tf
from network import Network

class CrossNet(Network):

    def __init__(self, inputs, channels=4, eps=1e-6, trainable=True):
        super(CrossNet, self).__init__(inputs, trainable=trainable)
        self.channels = channels
        self.eps = eps

        

    def build_model(self):

        
        # === magni ===

        (self.feed('magni')
             # 512x125x1
             .conv(5, 5, 16, 2, 2, padding='SAME', biased=False, relu=False, name='conv1')
             .batch_normalization(relu=False, name='conv1_bn')
             .leaky_relu(alpha=0.2, name='conv1_relu')
             # 256x64x16
             .conv(5, 5, 32, 2, 2, padding='SAME', biased=False, relu=False, name='conv2')
             .batch_normalization(relu=False, name='conv2_bn')
             .leaky_relu(alpha=0.2, name='conv2_relu')
             # 128x32x32
             .conv(5, 5, 64, 2, 2, padding='SAME', biased=False, relu=False, name='conv3')
             .batch_normalization(relu=False, name='conv3_bn')
             .leaky_relu(alpha=0.2, name='conv3_relu')
             # 64x16x64
             .conv(5, 5, 128, 2, 2, padding='SAME', biased=False, relu=False, name='conv4')
             .batch_normalization(relu=False, name='conv4_bn')
             .leaky_relu(alpha=0.2, name='conv4_relu')
             # 32x8x128
             .conv(5, 5, 256, 2, 2, padding='SAME', biased=False, relu=False, name='conv5')
             .batch_normalization(relu=False, name='conv5_bn')
             .leaky_relu(alpha=0.2, name='conv5_relu')
             # 16x4x256
             .conv(5, 5, 512, 2, 2, padding='SAME', biased=False, relu=False, name='conv6')
             .batch_normalization(relu=False, name='conv6_bn')
             .leaky_relu(alpha=0.2, name='conv6_relu'))
             # 8x2x512

        # === channels ===

        channel_list = []
        for c in range(self.channels):
            name1 = 'deconv1_{}'.format(c)
            (self.feed('conv6_relu')
                 .deconv(5, 5, 2, 2, self.layers['conv5_relu'].get_shape(), relu=False, name=name1)
                 .dropout(0.5, name=name1+'_drop')
                 .batch_normalization(relu=False, name=name1+'_bn')
                 .relu(name=name1+'_relu'))
            # 16x4x256
            (self.feed(name1+'_relu', 'conv5_relu')
                 .concat(axis=-1, name=name1+'_concat'))
            # 16x4x512
            name2 = 'deconv2_{}'.format(c)
            (self.feed(name1+'_concat')
                 .deconv(5, 5, 2, 2, self.layers['conv4_relu'].get_shape(), relu=False, name=name2)
                 .dropout(0.5, name=name2+'_drop')
                 .batch_normalization(relu=False, name=name2+'_bn')
                 .relu(name=name2+'_relu'))
            # 32x8x128
            (self.feed(name2+'_relu', 'conv4_relu')
                 .concat(axis=-1, name=name2+'_concat'))
            # 32x8x256
            name3 = 'deconv3_{}'.format(c)
            (self.feed(name2+'_concat')
                 .deconv(5, 5, 2, 2, self.layers['conv3_relu'].get_shape(), relu=False, name=name3)
                 .dropout(0.5, name=name3+'_drop')
                 .batch_normalization(relu=False, name=name3+'_bn')
                 .relu(name=name3+'_relu'))
            # 64x16x64
            (self.feed(name3+'_relu', 'conv3_relu')
                 .concat(axis=-1, name=name3+'_concat'))
            # 64x16x128
            name4 = 'deconv4_{}'.format(c)
            (self.feed(name3+'_concat')
                 .deconv(5, 5, 2, 2, self.layers['conv2_relu'].get_shape(), relu=False, name=name4)
                 .batch_normalization(relu=False, name=name4+'_bn')
                 .relu(name=name4+'_relu'))
            # 128x32x32
            (self.feed(name4+'_relu', 'conv2_relu')
                 .concat(axis=-1, name=name4+'_concat'))
            # 128x32x64
            name5 = 'deconv5_{}'.format(c)
            (self.feed(name4+'_concat')
                 .deconv(5, 5, 2, 2, self.layers['conv1_relu'].get_shape(), relu=False, name=name5)
                 .batch_normalization(relu=False, name=name5+'_bn')
                 .relu(name=name5+'_relu'))
            # 256x64x16
            (self.feed(name5+'_relu', 'conv1_relu')
                 .concat(axis=-1, name=name5+'_concat'))
            # 256x64x32
            name6 = 'deconv6_{}'.format(c)
            (self.feed(name5+'_concat')
                 .deconv(5, 5, 2, 2, self.layers['magni'].get_shape(), relu=False, name=name6)
                 .batch_normalization(relu=False, name=name6+'_bn')
                 .operation(lambda i, name: tf.nn.sigmoid(i, name=name), name=name6+'_sig'))
            # 512x128x1
            channel_list.append(self.layers[name6+'_sig'])

        self.output = channel_list

        self.summation = tf.add(tf.reduce_sum(tf.abs(tf.stack(self.output, axis=-1)), axis=-1), self.eps)

        return channel_list # mask
           

    def _post_process(self, input, final_layer):
        final_layer = tf.multiply(input, final_layer/self.summation)
        return final_layer


    def build_post_model(self):
        for c in range(len(self.output)):
            self.output[c] = self._post_process(self.layers['magni'], self.output[c])
        return self.output

