import argparse
import os
from os import path
import hashlib
import logging

from collections import OrderedDict
from train_val import Experiment


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Data and log directory
    parser.add_argument('-data_dir',
                        default='data/',
                        help='Root directory of data', type=str)
    parser.add_argument('-base_model_dir',
                        default='models/',
                        help='Root folder storing model runs', type=str)

    # Document encoder
    parser.add_argument('-model_size', default='base', type=str,
                        help='BERT model size')
    parser.add_argument('-query_mlp', default=False, action="store_true",
                        help='If true use an MLP for query vec o/w RNN.')

    # Memory parameters
    parser.add_argument('-mem_type', default='vanilla',
                        choices=['vanilla', 'learned', 'key_val'], help='Memory type.')
    parser.add_argument('-num_cells',
                        help='Number of memory cells', default=10, type=int)
    parser.add_argument('-mlp_size', default=300, type=int,
                        help='MLP hidden size used in the model.')
    parser.add_argument('-mem_size', default=300, type=int,
                        help='Memory cell size.')
    parser.add_argument('-usage_decay_rate', default=0.98,
                        type=float, help='Usage decay rate')
    parser.add_argument('-cumm', default='sum', choices=['sum', 'max'],
                        help='Score accumulation across memory cells.')

    # Training params
    parser.add_argument('-batch_size', help='Batch size', default=32, type=int)
    parser.add_argument('-feedback', default=False, action='store_true',
                        help='When true, do training on less data.')
    parser.add_argument('-dropout_rate', default=0.5, type=float, help='Dropout rate')
    parser.add_argument('-ent_loss', default=0.1, type=float,
                        help='Entity prediction loss on ent probabilities.')
    parser.add_argument('-max_epochs', default=100, type=int,
                        help='Maximum number of epochs')
    parser.add_argument('-max_num_stuck_epochs', default=15, type=int,
                        help='Maximum number of epochs without improvement.')
    parser.add_argument('-seed', default=0,
                        help='Random seed to get different runs', type=int)
    parser.add_argument('-init_lr', help="Initial learning rate",
                        default=1e-3, type=float)
    parser.add_argument('-eval', help="Evaluate model on GAP",
                        default=False, action="store_true")
    parser.add_argument('-slurm_id', help="Slurm ID", default=None, type=str)

    args = parser.parse_args()

    # Get model directory name
    opt_dict = OrderedDict()
    # Only include important options in hash computation
    imp_opts = ['model_size', 'mlp_size', 'query_mlp',
                'num_cells', 'mem_size', 'mem_type', 'usage_decay_rate',
                'cumm', 'ent_loss', 'dropout_rate', 'max_num_stuck_epochs',
                'batch_size', 'feedback', 'seed', 'init_lr']
    for key, val in vars(args).items():
        if key in imp_opts:
            opt_dict[key] = val

    str_repr = str(opt_dict.items())
    hash_idx = hashlib.md5(str_repr.encode("utf-8")).hexdigest()
    model_name = "petra_" + str(hash_idx)

    model_dir = path.join(args.base_model_dir, model_name)
    args.model_dir = model_dir
    best_model_dir = path.join(model_dir, 'best_models')
    args.best_model_dir = best_model_dir
    if not path.exists(model_dir):
        os.makedirs(model_dir)
    if not path.exists(best_model_dir):
        os.makedirs(best_model_dir)

    config_file = path.join(model_dir, 'config')
    with open(config_file, 'w') as f:
        for key, val in opt_dict.items():
            logging.info('%s: %s' % (key, val))
            f.write('%s: %s\n' % (key, val))

    try:
        Experiment(**vars(args))
    finally:
        pass


if __name__ == "__main__":
    main()
