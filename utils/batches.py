import sys
import numpy as np
from keras.applications import imagenet_utils
from .data import ListImageDataGenerator


def onehot_batches(sample_list, num_classes,
                   output_shape=(224, 224), batch_size=32):
    datagen = ListImageDataGenerator(
                    featurewise_center=False,             # Set input mean to 0 over the dataset
                    samplewise_center=False,              # Set each sample mean to 0
                    featurewise_std_normalization=False,  # Divide inputs by std of the dataset
                    samplewise_std_normalization=False,   # Divide each input by its std
                    zca_whitening=False,                  # Apply ZCA whitening
                    rotation_range=0,                     # Randomly rotate images in the range (degrees, 0 to 180)
                    width_shift_range=0.1,                # Randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.1,               # Randomly shift images vertically (fraction of total height)
                    horizontal_flip=True,                 # Randomly flip images
                    vertical_flip=False)                  # Randomly flip images
    preprocess = imagenet_utils.preprocess_input
    imgen = datagen.flow_from_list(sample_list, target_size=output_shape, batch_size=32,
                                   shuffle=True)
    print('Number of samples: {}'.format(imgen.num_samples))
    while True:
        x, y = next(imgen)
        embed_inds = np.tile(np.arange(num_classes), (y.shape[0], 1))
        x_hat = preprocess(x)
        yield [x_hat, embed_inds], y
