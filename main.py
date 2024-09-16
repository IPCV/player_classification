import copy
import json
import warnings
from argparse import ArgumentParser
from pathlib import Path

import torch
import wandb
from addict import Dict

from clustering import cluster
from test.backbone import test_backbone
from test.cluster import test_cluster
from test.cnn import test_cnn
from test.transformer import test_transformer
from train.backbone import train_backbone
from train.transformer import train_transformer_single_match, train_transformer_multiple_matches, find_lr
from util.misc import init_wandb, load_snv3_sequences, load_snv3_splits, load_snv2_splits, init_weights_dir
from util.misc import load_conf
from util.misc import set_random_seed


def train_and_test(conf, args, device, split=None):
    args = copy.deepcopy(args)

    set_random_seed(conf.optimization.random_seed)
    init_wandb(conf, args.architecture, mode=args.log_mode)

    if not args.only_test:
        if split is not None:
            wandb.run.name += f'_{split.name}'
        init_weights_dir(args)

        if args.architecture == 'transformer':
            train_transformer_single_match(split, conf, args, device)
        else:
            train_backbone(conf, args, device)

        args.checkpoint = args.weights_dir.joinpath('model.pth')

    if args.architecture == 'transformer':
        test_transformer(split.test, conf, args, device)
    else:
        test_backbone(conf, args, device)
    wandb.finish()


if __name__ == '__main__':
    parser = ArgumentParser(description='TransKit training')
    inputs_group = parser.add_mutually_exclusive_group()
    inputs_group.add_argument('-m', '--matches',
                              help='Path for a file containing the SoccerNetV2 training matches list (default: None)',
                              default=None, type=lambda p: Path(p))
    inputs_group.add_argument('-s', '--splits',
                              help='SoccerNetV2 splits filepath (default: None)',
                              default=None, type=lambda p: Path(p))
    inputs_group.add_argument('--sequences',
                              help='JSON file containing SoccerNetV3 sequences ordered by half matches (default: None)',
                              default=None, type=lambda p: Path(p))
    parser.add_argument('-a', '--architecture', required=False,
                        help='Architecture component to train/test (default: transformer)',
                        default='transformer', choices=['backbone', 'cnn', 'transformer'])
    parser.add_argument('-c', '--conf', required=False,
                        help='JSON configuration filepath (default: config/transformer.json)',
                        default="config/transformer.json", type=lambda p: Path(p))
    parser.add_argument('-d', '--dataset_path', required=False,
                        help='Path for SoccerNet dataset (default: data/soccernet)',
                        default="data/soccernet", type=lambda p: Path(p))
    weights_group = parser.add_mutually_exclusive_group()
    weights_group.add_argument('-b', '--backbone_weights', required=False,
                               help='Backbone weights filepath (default: None)',
                               default=None, type=lambda p: Path(p))
    weights_group.add_argument('-t', '--transformer_weights', required=False,
                               help='Transformer weights filepath (default: None)',
                               default=None, type=lambda p: Path(p))
    parser.add_argument('--only_test', required=False,
                        help='Only perform test (default: False)',
                        action='store_true')
    parser.add_argument('--lr_finder', required=False,
                        help='Learning rate finder (default: False)',
                        action='store_true')
    parser.add_argument('--save_predictions', required=False,
                        help='Filename for saving predictions (default: None)',
                        default=None, type=str)
    parser.add_argument('--cluster', required=False,
                        help='Cluster the players from SoccerNet matches (default: False)',
                        action='store_true')
    parser.add_argument('-k', '--kit_clusters_csv', required=False,
                        help='Kit uniform clusters CSV filepath for the SoccerNet matches (default: None)',
                        default=None, type=lambda p: Path(p))
    parser.add_argument('--weights_dir', required=False,
                        help='Weights directory path (default: weights)',
                        default='weights', type=lambda p: Path(p))
    parser.add_argument('--log_mode', required=False,
                        help='Log mode [online, offline, disabled] (default: online)',
                        default='online', type=str)
    parser.add_argument('--log_freq', required=False,
                        help='Results log frequency (default: 20)',
                        default=20, type=int)
    parser.add_argument('--checkpoint', required=False,
                        help='Training checkpoint filepath (default: None)',
                        default=None, type=lambda p: Path(p))
    parser.add_argument('--num_loading_threads', required=False,
                        help='Number of data loading threads (default: 12)',
                        default=12, type=int)
    parser.add_argument('--num_workers', required=False,
                        help='Number of workers for PyTorch dataloaders (default: 2)',
                        default=2, type=int)
    args = parser.parse_args()

    args.is_soccernet_v3 = args.sequences is not None
    args.individual_matches = True
    if args.is_soccernet_v3:
        args.sequences = load_snv3_sequences(args.dataset_path, args.sequences)
    elif args.matches is not None:
        with args.matches.open() as f:
            args.matches = [args.dataset_path.joinpath(m) for m in f.read().splitlines()]
    else:
        with args.splits.open() as json_file:
            args.splits = Dict(json.load(json_file))
            for s in ['train', 'valid', 'test']:
                args.splits[s] = [args.dataset_path.joinpath(m) for m in args.splits[s]]
        args.individual_matches = False

    conf = load_conf(args.conf, args.architecture, args.backbone_weights, args.transformer_weights)

    device = torch.device("cuda")

    set_random_seed(conf.optimization.random_seed)

    if args.cluster:
        if not args.only_test:
            cluster(conf, args, device)
        test_cluster(conf, args, device)
    elif args.architecture == 'backbone':
        train_and_test(conf, args, device)
    elif args.architecture == 'cnn':
        if not args.only_test:
            warnings.warn("Warning: Training CNNs on SoccerNetV2 is not supported for CNNs")

        if args.is_soccernet_v3:
            raise Exception('Training/Testing CNNs on SoccerNetV3 is not implemented for CNNs')

        wandb.init(mode='disabled')
        for match_path in args.matches:
            test_cnn(match_path, conf, args, device)
    elif args.architecture == 'transformer':
        if args.individual_matches:
            if args.is_soccernet_v3:
                splits = load_snv3_splits(args.sequences, conf.optimization.split_train_ratio)
                for split in splits:
                    train_and_test(conf, args, device, split)
            else:
                for match_path in args.matches:
                    try:
                        split = load_snv2_splits(match_path, conf)
                        train_and_test(conf, args, device, split)
                    except:
                        print(match_path)
                        wandb.finish()
        elif not args.lr_finder:
            if not args.only_test:
                init_wandb(conf, args.architecture, mode=args.log_mode)
                init_weights_dir(args)

                train_transformer_multiple_matches(conf, args, device)

            args.checkpoint = args.weights_dir.joinpath('model.pth')
            for match_path in args.splits.test:
                test_transformer(match_path, conf, args, device)
        else:
            init_wandb(conf, args.architecture, mode=args.log_mode)
            init_weights_dir(args)

            find_lr(conf, args, device)
