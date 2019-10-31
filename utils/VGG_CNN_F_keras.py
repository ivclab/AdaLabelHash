from __future__ import print_function
from keras.layers import (Input, Dense, Conv2D, MaxPooling2D, ZeroPadding2D,
                          Flatten, Dropout, GlobalAveragePooling2D,
                          GlobalMaxPooling2D, Dropout, Activation)
from keras.engine import Layer, Model
from keras import backend as K
import numpy as np


class LRN(Layer):
    def __init__(self, alpha=0.0001,k=1,beta=0.75,n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN, self).__init__(**kwargs)

    def call(self, x, mask=None):
        b, r, c, ch = x.shape
        half_n = self.n // 2    # half the local region
        input_sqr = K.square(x) # square the input

        input_sqr = K.spatial_2d_padding(input_sqr, padding=((0, 0), (half_n, half_n)),
                                         data_format='channels_first')
        scale = self.k                   # offset for the scale
        norm_alpha = self.alpha / self.n # normalized alpha
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, :, :, i:i+int(ch)]
        scale = scale ** self.beta
        x = x / scale
        return x

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def VGG_CNN_F(arch_mode='full', weights='imagenet',
              input_shape=(224, 224, 3), pooling=None, classes=1000):
    """ This function constructs VGG_CNN_F and is based on
        https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py

        Arguments:
            arch_mode: {'full', 'rmlast', 'notop'}
    """

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and arch_mode == 'full' and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `arch_mode`'
                         ' as `full`, `classes` should be 1000')

    img_input = Input(shape=input_shape)
    x = Conv2D(64, (11, 11), strides=(4, 4), activation='relu', name='conv1')(img_input)  # conv1
    x = LRN(n=5, alpha=0.0005, beta=0.75, k=2, name='norm1')(x)                           # norm1
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(x)   # pool1

    x = ZeroPadding2D(padding=(2, 2))(x)
    x = Conv2D(256, (5, 5), strides=(1, 1), activation='relu', name='conv2')(x)           # conv2
    x = LRN(n=5, alpha=0.0005, beta=0.75, k=2, name='norm2')(x)                           # norm2
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool2')(x)                   # pool2

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', name='conv3')(x)           # conv3

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', name='conv4')(x)           # conv4

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', name='conv5')(x)           # conv5
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)                   # pool5

    if arch_mode == 'notop':
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
    else:
        x = Flatten()(x)
        x = Dense(4096, activation='relu', name='fc6')(x)                                 # fc6
        x = Dropout(0.5)(x)                                                               # drop6
        x = Dense(4096, activation='relu', name='fc7')(x)                                 # fc7
        x = Dropout(0.5)(x)                                                               # drop7

        if arch_mode == 'full':
            x = Dense(classes, name='fc8')(x)                                             # fc8
            x = Activation('softmax', name='prob')(x)                                     # prob
        elif arch_mode != 'rmlast':
            raise ValueError('arch_mode: {} is not supported.'.format(arch_mode))

    inputs = img_input

    # Create model.
    model = Model(inputs, x, name='vgg_cnn_f')

    # load weights
    if weights == 'imagenet':
        if arch_mode == 'full':
            weights_path = 'init_weights/vgg_cnn_f.h5'
        elif arch_mode == 'rmlast':
            weights_path = 'init_weights/vgg_cnn_f_rmlast.h5'
        elif arch_mode == 'notop':
            weights_path = 'init_weights/vgg_cnn_f_notop.h5'
        model.load_weights(weights_path)

        if K.backend() == 'theano':
            raise NotImplemented('Support for loading imagenet weights with '
                                 'theano backend is not implemented yet!')

        if K.image_data_format() == 'channels_first':
            raise NotImplemented('Support for loading imagenet weights with '
                                 'channels_first image data format is not implemented yet!')
    elif weights is not None:
        model.load_weights(weights)
    return model
