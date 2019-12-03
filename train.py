import numpy as np
import sys
import os
import argparse
import importlib
import tensorflow as tf
import keras.backend as K
from pathlib2 import Path
from keras import optimizers
from keras.models import load_model
from keras.engine import Model
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from utils.VGG_CNN_F_keras import LRN
from utils.configs import parse_config
from utils.models import construct_adalabelhash
from utils.losses import construct_triplet_loss
from utils.data import load_sample_list
from utils.batches import  onehot_batches


def parse_args():
    parser = argparse.ArgumentParser(description='In train.py')

    parser.add_argument('--gpu-id', type=str, required=False, default='0',
                        help='GPU ids to run')
    parser.add_argument('--exp-dir', type=str, required=True,
                        help='Experiment directory')
    parser.add_argument('--sample-file', type=str, required=True,
                        help='List of samples for training')
    parser.add_argument('--code-len', type=int, required=True,
                        help='Length of hash codes')
    parser.add_argument('--num-classes', type=int, required=True,
                        help='Number of classes')
    parser.add_argument('--config-file', type=str, required=True,
                        help='Configuration file')
    parser.add_argument('--config-opt', type=str, required=True,
                        help='Target option in the configuration file')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def run_train(args, optim_args):

    ## Initial preparation
    model_dir = os.path.join(args.exp_dir, 'models', args.config_opt)
    # os.makedirs(model_dir, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_check_point = ModelCheckpoint(os.path.join(model_dir, '{epoch:03d}.h5'),
                                        period=args.save_period)
    csv_logger        = CSVLogger(os.path.join(model_dir, 'log.csv'))
    early_stopping    = EarlyStopping(monitor='loss', patience=args.patience)

    ## Tensorflow settings
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    ## Model construction
    optimizer = getattr(importlib.import_module('keras.optimizers'),
                        optim_args[0])(**optim_args[1])

    model = construct_adalabelhash(args.code_len, (224, 224), args.num_classes,
                                   sim_name='innerprod')

    if args.pretrain != '':
        print('Load pretrained model: {}'.format(args.pretrain))
        model.load_weights(args.pretrain, by_name=True)
    model.compile(loss=construct_triplet_loss(num_classes=args.num_classes, k=args.k),
                  optimizer=optimizer)

    ## Construct dataloader
    sample_list = load_sample_list(args.sample_file)
    datagen     = onehot_batches(sample_list, args.num_classes,
                                 output_shape=(224, 224), batch_size=args.batch_size)
    num_iters   = int(np.ceil(float(len(sample_list)) / float(args.batch_size)))

    # ## Run training iterations
    model.fit_generator(datagen, num_iters, epochs=args.max_epochs, verbose=1,
                        callbacks=[csv_logger, model_check_point, early_stopping])
    model.save(os.path.join(model_dir, 'model.h5'))
    return


def main():
    args = parse_args()
    print('Arguments: {}'.format(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    optim_args, model_args = parse_config(args.config_file, args.config_opt)

    # Mergs model_args with args
    for k, v in model_args.items():
        setattr(args, k, v)

    run_train(args, optim_args)
    return


if __name__ == "__main__":
    main()
