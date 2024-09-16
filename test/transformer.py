from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm
from IO.soccernetv2.match import PlayerPatchesDataset as SNV2Dataset, MatchDataset
from IO.soccernetv3.sequences import PlayerPatchesDataset as SNV3Dataset

from models.transkit import build
from util.data import CollateFrames, make_seq_mask, unmask_flat_sequences

__all__ = ['test_transformer']


def evaluate(model, loader, criterion, device) -> Tuple[float, np.ndarray]:
    msg = 'Testing {}/{}'

    model.eval()
    predictions, targets, unflatten_predictions = [], [], []
    with torch.no_grad():
        for batch_idx, (players, coords, labels, lengths) in (pbar := tqdm(enumerate(loader))):
            players, coords, labels = players.to(device), coords.to(device), labels.to(device)
            seq_mask = make_seq_mask(lengths).to(device)
            outputs = model(players, coords, src_key_padding_mask=seq_mask)
            prediction = outputs.view(-1, model.transformer.num_classes).argmax(dim=1)
            predictions.append(prediction)
            unflatten_predictions.extend(unmask_flat_sequences(prediction, lengths))
            targets.append(labels)
            pbar.set_description(msg.format(batch_idx + 1, len(loader)))
            pbar.refresh()
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    accuracy = criterion(predictions, targets)
    if wandb.run is not None:
        wandb.log({'Test/test_accuracy': accuracy})
    print(f"Test accuracy = {accuracy}")

    return accuracy, np.concatenate(unflatten_predictions)


def test_transformer(test_match: Union[Path, MatchDataset], conf, args, device):
    soccernet_dataset = SNV3Dataset if args.is_soccernet_v3 else SNV2Dataset

    model = build(conf).to(device)

    accuracy_metric = Accuracy(task="multiclass",
                               num_classes=model.transformer.num_classes,
                               ignore_index=conf.data.padding_token).to(device)

    test_dataset = soccernet_dataset(test_match,
                                     load_gt=True,
                                     use_background=conf.backbone.background,
                                     num_processes=args.num_loading_threads)
    
    test_loader = DataLoader(dataset=test_dataset,
                             collate_fn=CollateFrames(num_data_outputs=test_dataset.num_data_outputs,
                                                      num_classes=conf.transformer.model.num_classes,
                                                      padding_token=conf.data.padding_token),
                             shuffle=False,
                             batch_size=conf.optimization.batch_size,
                             num_workers=args.num_workers)

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    accuracy, predictions = evaluate(model, test_loader, accuracy_metric, device)

    if args.save_predictions:
        test_dataset.save_predictions(predictions, filename=args.save_predictions)
