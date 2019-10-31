import os
import copy
from configparser import SafeConfigParser


def parse_config(filepath, option):
    parser = SafeConfigParser()
    parser.read(filepath)

    # Loading optimizer arguments
    print('== '+option+'.optim ==')
    optim_dict = {}
    section_name = option+'.optim'
    for key in parser.options(section_name):
        if key == 'optimizer':
            optim_name = parser.get(section_name, key)
        else:
            if key == 'nesterov' or key == 'amsgrad':
                value = parser.getboolean(section_name, key)
            else:
                value = parser.getfloat(section_name, key)
            optim_dict[key] = value
    optim_args = (optim_name, optim_dict)
    print('Optim name: {}'.format(optim_name))
    for k, v in optim_dict.items():
        print('{:<12} --> {}'.format(k, v))

    # Loading model arguments
    print('== '+option+'.model ==')
    int_default   = dict(batch_size=32, save_period=20,
                         patience=5, max_epochs=100)
    float_default = dict(k=1.0)
    str_default   = dict(pretrain='')
    model_args = {}
    model_args.update(int_default)
    model_args.update(float_default)
    model_args.update(str_default)
    section_name = option+'.model'
    if parser.has_section(section_name):
        for key in parser.options(section_name):
            if key in int_default:
                value = parser.getint(section_name, key)
            elif key in float_default:
                value = parser.getfloat(section_name, key)
            else:
                value = parser.get(section_name, key)
            model_args[key] = value
    for k, v in model_args.items():
        print('{:<12} --> {}'.format(k, v))
    return optim_args, model_args
