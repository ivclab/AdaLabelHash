import sys
import os
import re
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
from utils.VGG_CNN_F_keras import LRN
from utils.models import predict_image_codes
from utils.data import load_sample_list


def parse_args():
    parser = argparse.ArgumentParser(description='In test.py')

    parser.add_argument('--gpu-id', type=str, required=False, default='0',
                        help='GPU ids to run')
    parser.add_argument('--exp-dir', type=str, required=True,
                        help='Experiment directory')
    parser.add_argument('--code-len', type=int, required=True,
                        help='Length of hash codes')
    parser.add_argument('--sample-files', type=str, required=True, action='append',
                        help='List of samples for training')
    parser.add_argument('--num-classes', type=int, required=True,
                        help='Number of classes')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def run_test(args):
    ## Initial preparation
    result_dir = os.path.join(args.exp_dir, 'models', '{}bits'.format(args.code_len))
    model_path = os.path.join(result_dir, 'model.h5')

    ## Tensorflow config settings
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    # ## Run testing iterations
    model = load_model(model_path, compile=False, custom_objects={'LRN': LRN})
    for sample_file in args.sample_files:
        sample_list = load_sample_list(sample_file)
        image_paths, labels = zip(*sample_list)
        output_name = re.split('_', os.path.basename(os.path.splitext(sample_file)[0]))[-1]
        output_path = os.path.join(result_dir, output_name)
        x = predict_image_codes(model, sample_list)
        y = np.array(labels, dtype=np.float32)
        np.save(output_path+'_x.npy', x)
        np.save(output_path+'_y.npy', y)
        print(('Store predicts, labels with shape {}, {} '
               'to {}').format(x.shape, y.shape, output_path+'{_x, _y}.npy'))
    return


def main():
    args = parse_args()
    print('Arguments: {}'.format(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    run_test(args)
    return


if __name__ == '__main__':
    main()
