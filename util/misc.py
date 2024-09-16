import json
import os
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import wandb
from addict import Dict
from pytorch_metric_learning import testers

from IO.soccernetv2.match import MatchDataset
from kitman.snv2 import FrameIndex, shorten_name

from sklearn.model_selection import train_test_split
import shutil
import subprocess


def set_random_seed(seed=42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def load_conf(json_conf: Path, architecture: str = None, backbone_weights: Path = None,
              transformer_weights: Path = None) -> Dict:
    with json_conf.open() as json_file:
        conf = Dict(json.load(json_file))

    if conf.get('cnn', None) is None:

        conf.backbone.background = conf.data.use_background

        # Player coords configuration
        conf.backbone.model.use_player_coords = conf.data.use_player_coords
        conf.transformer.model.dim = conf.backbone.model.dim
        if conf.data.use_player_coords:
            conf.transformer.model.dim += 2

        # Positional, patch, modality encoding configuration
        conf.backbone.model.use_positional_encoding = conf.transformer.model.get('use_positional_encoding', False)
        conf.backbone.model.use_patch_encoding = conf.transformer.model.get('use_patch_encoding', False)
        conf.backbone.model.use_modality_encoding = conf.transformer.model.get('use_modality_encoding', False)

        # Number of classes
        conf.transformer.model.num_classes = conf.data.num_classes
        conf.clustering.num_clusters = conf.data.num_classes

        if len(conf.backbone.metric):
            conf.clustering.distance = conf.backbone.metric.distance.lower()

        if conf.clustering.get('separate_player_clusters', None) is None:
            conf.clustering.separate_player_clusters = False

        conf.transformer.optimization['player_count'] = conf.transformer.optimization.get('player_count', None)
        if conf.transformer.optimization.player_count:
            max_count_per_class = conf.transformer.optimization.player_count.max_count_per_class
            max_count_per_class = {int(c): max_count for c, max_count in max_count_per_class.items()}
            conf.transformer.optimization.player_count.max_count_per_class = max_count_per_class

        if conf.transformer.optimization.early_stopping_patience is None:
            conf.transformer.optimization.early_stopping_patience = conf.transformer.optimization.num_epochs

        conf.transformer.optimization['freeze_backbone'] = conf.transformer.optimization.get('freeze_backbone', True)

        # Batch, Sequence shape configuration
        conf.backbone.model.batch_first = conf.transformer.model.batch_first

        conf.backbone.weights = backbone_weights
        conf.transformer.weights = transformer_weights

    conf.optimization = conf[architecture].optimization
    return conf


def get_git_revision_hash() -> str:
    try:
        command_output = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT)
        return command_output.decode('ascii').strip()
    except:
        return None


def init_wandb(conf: Dict, component: str, name: str = None, mode: str = None) -> None:
    wandb.login()
    if component == 'backbone':
        wandb_conf = {"learning_rate": conf.optimization.learning_rate,
                      "architecture": conf.backbone.model.name,
                      "dim": conf.backbone.model.dim,
                      "batch_size": conf.optimization.batch_size,
                      "loss": conf.backbone.metric.loss,
                      "distance": conf.backbone.metric.distance,
                      "reducer": conf.backbone.metric.reducer,
                      "epochs": conf.optimization.num_epochs,
                      "miner": conf.backbone.metric.miner,
                      "use_background": conf.data.use_background,
                      }
    else:
        wandb_conf = {"learning_rate": conf.optimization.learning_rate,
                      "dim": conf.transformer.model.dim,
                      "num_heads": conf.transformer.model.num_heads,
                      "num_encoder_layers": conf.transformer.model.num_encoder_layers,
                      "num_classes": conf.transformer.model.num_classes,
                      "dim_feedforward": conf.transformer.model.dim_feedforward,
                      "dropout": conf.transformer.model.dropout,
                      "norm_first": conf.transformer.model.norm_first,
                      "batch_first": conf.transformer.model.batch_first,
                      "use_player_coords": conf.data.use_player_coords,
                      "use_groundtruth": conf.data.use_groundtruth,
                      "use_background": conf.data.use_background,
                      "backbone": conf.backbone.model.name,
                      "backbone_loss": conf.backbone.metric.loss,
                      "distance": conf.backbone.metric.distance,
                      "reducer": conf.backbone.metric.reducer,
                      "miner": conf.backbone.metric.miner
                      }
    git_commit = get_git_revision_hash()
    if git_commit:
        wandb_conf['git_revision_hash'] = git_commit

    if conf.backbone.metric.loss == 'NTXentLoss' or conf.backbone.metric.loss == 'InfoNCE':
        wandb_conf["temperature"] = conf.backbone.metric.temperature
    wandb.init(config=wandb_conf,
               project=f'player_classification_{component}',
               name=name,
               mode=mode)


def init_weights_dir(args):
    weights_dir = args.weights_dir.joinpath(wandb.run.name)
    weights_dir.mkdir(parents=True, exist_ok=True)
    args.weights_dir = weights_dir
    shutil.copyfile(args.conf, args.weights_dir.joinpath('config.json'))


def load_snv3_sequences(dataset: Path, snv3_matches: Path):
    half_matches = []
    with snv3_matches.open() as json_file:
        for m in json.load(json_file):
            for half in ['first_half', 'second_half']:
                if half in m:
                    half_matches.append([dataset.joinpath('tracking', p) for p in m[half]])
    return half_matches


def load_snv3_splits(snv3_half_matches: List[List[Path]], split_train_ratio: float = 0.8):
    splits = []

    for idx, (first_half, second_half) in enumerate(zip(snv3_half_matches[::2],
                                                        snv3_half_matches[1::2])):
        num_sequences = len(first_half)
        indices = np.arange(num_sequences)
        np.random.shuffle(indices)

        num_train_samples = int(split_train_ratio * num_sequences)
        train_indices = indices[:num_train_samples]
        valid_indices = indices[num_train_samples:]

        splits.append(Dict({'name': f'seq_{idx}',
                            'train': [first_half[i] for i in train_indices],
                            'valid': [first_half[i] for i in valid_indices],
                            'test': second_half}))
    return splits


def load_snv2_splits(match_path: Path, conf):
    # Clustered predictions dataset
    cluster_method = 'gk_separation' if conf.clustering.separate_goalkeepers else 'spectral'
    dataset = MatchDataset(match_path,
                           load_predictions=True,
                           use_background=conf.backbone.background,
                           cluster_method=cluster_method)

    test_dataset = MatchDataset(match_path,
                                load_gt=True,
                                use_background=conf.backbone.background)

    frame_ids = []
    for half in range(2):
        for frame_id in dataset.valid_frames_ids[half]:
            if frame_id not in test_dataset.valid_frames_ids[half]:
                frame_ids.append(FrameIndex(half, frame_id))

    # Split the indices into training and validation sets
    train_indices, validation_indices = train_test_split(np.arange(len(frame_ids)),
                                                         test_size=1 - conf.optimization.split_train_ratio,
                                                         random_state=conf.optimization.random_seed)

    train_frame_ids = [frame_ids[i] for i in train_indices]
    validation_frame_ids = [frame_ids[i] for i in validation_indices]

    train_split = dataset.split_by(train_frame_ids)
    valid_split = dataset.split_by(validation_frame_ids)

    return Dict({'name': shorten_name(match_path),
                 'train': train_split,
                 'valid': valid_split,
                 'test': test_dataset})
