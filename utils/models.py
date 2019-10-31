import sys
import numpy as np
import keras.backend as K
from keras.layers import Dense, Embedding, Input, Activation, Lambda
from keras.engine import Model
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from PIL import Image as pil_image
from .VGG_CNN_F_keras import VGG_CNN_F, LRN
from sklearn.cluster import KMeans


def innerprod_similarity(codes):
    res = K.dot(codes[0], K.transpose(codes[1][0, ...]))
    return res


def l2norm_innerprod_similarity():
    x0_norm = tf.nn.l2_normalize(x[0], 1)
    x1_norm = tf.nn.l2_normalize(x[1][0, ...], 1)
    res = K.dot(x0_norm, K.transpose(x1_norm))
    return res


def pair_similarity_shape(input_shape):
    shape = list(input_shape)
    shape[-1] = shape[0]
    return tuple(shape)


def construct_image_coder(code_len, in_shape):
    base_model  = VGG_CNN_F(arch_mode='rmlast', weights='imagenet', input_shape=in_shape+(3,))
    image_codes = Dense(code_len, name='hash_layer')(base_model.output)
    image_codes = Activation('tanh', name='image_codes')(image_codes)
    return Model(base_model.input, image_codes)


def construct_class_coder(code_len, num_classes):
    embed_inds  = Input(shape=(num_classes,), dtype='int32', name='embed_inds')
    class_codes = Embedding(num_classes, code_len, input_length=num_classes,
                            name='class_embedding')(embed_inds)
    class_codes = Activation('tanh', name='class_codes')(class_codes)
    return Model(embed_inds, class_codes)


def predict_image_codes(model, sample_list, input_shape=(224, 224)):
    image_coder = Model(inputs=[model.get_layer('input_1').input],
                        outputs=[model.get_layer('image_codes').output])
    batch_size = 128
    num_samples = len(sample_list)
    preprocess = imagenet_utils.preprocess_input
    count = 0
    image_paths, labels = zip(*sample_list)
    image_codes = []

    def load_and_resize_image(image_path, output_shape):
        image = pil_image.open(image_path)
        image = image.resize(output_shape, pil_image.BILINEAR)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = img_to_array(image)
        return [image]

    while count < num_samples:
        sys.stdout.write('prog: {}/{} ...        \r'.format(count, num_samples))
        sys.stdout.flush()
        cur_batch = np.min([batch_size, num_samples-count])
        cur_x = map(lambda x: load_and_resize_image(x, input_shape), image_paths[count:count+cur_batch])
        cur_x = preprocess(np.vstack(cur_x))
        cur_codes = image_coder.predict([cur_x])
        image_codes.append(cur_codes)
        count += cur_batch
    print('\nProcess %d samples' % num_samples)
    return np.vstack(image_codes)


def predict_class_codes(model, num_classes):
    class_coder = Model(inputs=[model.get_layer('embed_inds').input],
                        outputs=[model.get_layer('class_codes').output])
    embed_inds = np.tile(np.arange(num_classes), (1, 1))
    class_codes = class_coder.predict([embed_inds])
    class_codes = np.squeeze(class_codes)
    print('Shape of class_codes: {}'.format(class_codes.shape))
    return class_codes


def construct_adalabelhash(code_len, in_shape, num_classes, sim_name='innerprod'):
    """Construct the structure of ResHash Network

    # Arguments:
        in_shape: Input shape of images
        num_classes: Number of classes for supervisions
        code_len: Length of hash codes
        sim_name: Method for measuring the code similarities, ('innerprod', 'l2norm_innerprod')
    """
    sim_methods = {'innerprod':        innerprod_similarity,
                   'l2norm_innerprod': l2norm_innerprod_similarity}
    image_codes = construct_image_coder(code_len, in_shape)
    class_codes = construct_class_coder(code_len, num_classes)
    distance = Lambda(sim_methods[sim_name],
                      output_shape=pair_similarity_shape)([image_codes.output, class_codes.output])
    model = Model(inputs=[image_codes.input, class_codes.input], outputs=distance)
    return model
