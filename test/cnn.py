from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.models as models
import wandb
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm

from IO.soccernetv2.match import PlayerPatchesDataset
from util.data import CollateFrames

__all__ = ['test_cnn']


def evaluate(model, loader, criterion, device) -> Tuple[float, np.ndarray]:
    msg = 'Testing {}/{}'

    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch_idx, (players, _, labels, lengths) in (pbar := tqdm(enumerate(loader))):
            players, labels = players.to(device), labels.to(device)
            largest_size = players.shape[1]

            for j, k in enumerate(lengths):
                output = model(players[j, :k, ...])
                prediction = output.view(-1, 5).argmax(dim=1)
                predictions.append(prediction)

                start = j * largest_size
                targets.append(labels[start:start + k])

            pbar.set_description(msg.format(batch_idx + 1, len(loader)))
            pbar.refresh()
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    accuracy = criterion(predictions, targets)
    wandb.log({'Test/test_accuracy': accuracy})
    print(f"Test accuracy = {accuracy}")
    return accuracy, predictions.cpu().detach().numpy()


def load_model(conf):
    cnn = models.mobilenet_v3_large(pretrained=False)
    cnn.classifier._modules['3'] = nn.Linear(1280, conf.data.num_classes)
    return cnn


def test_cnn(match_path: Path, conf, args, device):
    model = load_model(conf).to(device)

    accuracy_metric = Accuracy(task="multiclass",
                               num_classes=conf.data.num_classes,
                               ignore_index=conf.data.padding_token).to(device)

    test_dataset = PlayerPatchesDataset(match_path,
                                        load_gt=True,
                                        use_background=conf.backbone.background,
                                        num_processes=args.num_loading_threads)
    test_loader = DataLoader(dataset=test_dataset,
                             collate_fn=CollateFrames(num_data_outputs=test_dataset.num_data_outputs,
                                                      num_classes=conf.data.num_classes,
                                                      padding_token=conf.data.padding_token),
                             shuffle=False,
                             batch_size=conf.optimization.batch_size,
                             num_workers=args.num_workers)

    weights = match_path.joinpath('player_labeling/weights/mobilenet_v3_large/initial_model.pth.tar')
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['state_dict'])

    accuracy, predictions = evaluate(model, test_loader, accuracy_metric, device)

    if args.save_predictions:
        test_dataset.save_predictions(predictions, filename=args.save_predictions)
