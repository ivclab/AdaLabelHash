import sys
import os
import re
import numpy as np
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator, Iterator, load_img
from keras.preprocessing.image import img_to_array, array_to_img


class ListImageDataGenerator(ImageDataGenerator):
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None):
        super(ListImageDataGenerator, self).__init__(featurewise_center=featurewise_center,
                                                     samplewise_center=samplewise_center,
                                                     featurewise_std_normalization=featurewise_std_normalization,
                                                     samplewise_std_normalization=samplewise_std_normalization,
                                                     zca_whitening=zca_whitening,
                                                     zca_epsilon=zca_epsilon,
                                                     rotation_range=rotation_range,
                                                     width_shift_range=width_shift_range,
                                                     height_shift_range=height_shift_range,
                                                     shear_range=shear_range,
                                                     zoom_range=zoom_range,
                                                     channel_shift_range=channel_shift_range,
                                                     fill_mode=fill_mode,
                                                     cval=cval,
                                                     horizontal_flip=horizontal_flip,
                                                     vertical_flip=vertical_flip,
                                                     rescale=rescale,
                                                     preprocessing_function=preprocessing_function,
                                                     data_format=data_format)


    def flow_from_list(self, sample_list,
                       target_size=(256, 256), color_mode='rgb',
                       batch_size=32, shuffle=True, seed=None,
                       save_to_dir=None,
                       save_prefix='',
                       save_format='png'):
        return ListIterator(
                sample_list, self,
                target_size=target_size, color_mode=color_mode,
                data_format=self.data_format,
                batch_size=batch_size, shuffle=shuffle, seed=seed,
                save_to_dir=save_to_dir,
                save_prefix=save_prefix,
                save_format=save_format)


class ListIterator(Iterator):
    """Iterator capable of reading images from sample list

    # Arguments
        sample_list: List of samples consist of (image_path, labels)
        image_data_generator: Instance of `ImageDataGenerator`
            to user for random transformations and normalization.
        target_size: tuple of integers, dimensions to reize image images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).

    """
    def __init__(self, sample_list, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png'):
        if data_format is None:
            data_format = K.image_data_format()
        self.sample_list = sample_list
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm'}
        self.num_samples = len(self.sample_list)

        super(ListIterator, self).__init__(self.num_samples, batch_size, shuffle, seed)
        return

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        batch_y = list()
        grayscale = self.color_mode == 'grayscale'
        for i, j in enumerate(index_array):
            sample = self.sample_list[j]
            img = load_img(sample[0],
                           grayscale=grayscale,
                           target_size=self.target_size)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_y.append(sample[1])
        batch_y = np.array(batch_y, dtype=K.floatx())

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


def load_sample_list(filepath):
    sample_list = []
    with open(filepath, 'r') as f:
        line = f.readline()
        while True:
            line = f.readline().strip()
            if line == '':
                break
            tokens = list(filter(None, re.split(',', line)))
            sample = (tokens[0], tuple([float(x) for x in tokens[1:]]))
            sample_list.append(sample)
    print('Load {} samples from {}'.format(len(sample_list), filepath))
    return sample_list
