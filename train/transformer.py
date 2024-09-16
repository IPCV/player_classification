from typing import List, Tuple

import numpy as np
import torch
import torch.optim as optim
import wandb
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from IO.soccernetv2.match import (PlayerPatchesDataset, CombineMatchesDataset)
from models.transkit import build
from util.data import CollateFrames, make_seq_mask
from util.loss import SequenceCountPenalty, PermutationInvariant
from util.transforms import player_transform, RandomCoordsHorizontalFlip

__all__ = ['train_transformer_single_match', 'train_transformer_multiple_matches']


def train(model, loader, optimizer, criterion, penalty, permute, device, epoch, log_freq=20) -> Tuple[
    List[float], List[float]]:
    msg = 'Epoch {} Iteration {}: Loss = {}'

    model.train()
    log_freq = min(len(loader), log_freq)

    batch_losses, batch_sizes = [], []
    for batch_idx, batch in (batches_bar := tqdm(enumerate(loader))):
        if loader.dataset.num_data_outputs == 3:
            players, coords, labels, lengths = batch
        else:
            players, coords, labels, match_ids, lengths = batch

        players, coords, labels = players.to(device), coords.to(device), labels.to(device)
        seq_mask = make_seq_mask(lengths).to(device)

        optimizer.zero_grad()
        outputs = model(players, coords, src_key_padding_mask=seq_mask)

        if permute is not None:
            labels = permute(labels, outputs, match_ids, seq_mask)

        criterion_loss = criterion(outputs.view(-1, model.transformer.num_classes), labels)
        penalty_loss = penalty(outputs, seq_mask) if penalty is not None else 0

        loss = criterion_loss + penalty_loss
        loss.backward()
        optimizer.step()

        metrics = {"train/batch/epoch": epoch,
                   "train/batch/iteration": batch_idx,
                   "train/batch/criterion_loss": criterion_loss,
                   "train/batch/penalty_loss": penalty_loss,
                   "train/batch/train_loss": loss}

        batches_bar.set_description(msg.format(epoch, batch_idx + 1, loss))
        batches_bar.refresh()
        if wandb.run is not None and (batch_idx + 1) % log_freq == 0:
            wandb.log(metrics)

        batch_losses.append(float(loss))
        batch_sizes.append(float(lengths.sum()))
    return batch_losses, batch_sizes


def evaluate(model, loader, criterion, penalty, permute, device) -> Tuple[float, List[float], List[float]]:
    msg = 'Validation {}/{}: Loss = {}'

    model.eval()
    acc_loss, acc_penalty_loss, acc_num_batches = 0., 0., 0
    batch_losses, batch_sizes = [], []
    with torch.no_grad():
        for batch_idx, batch in (pbar := tqdm(enumerate(loader))):
            if loader.dataset.num_data_outputs == 3:
                players, coords, labels, lengths = batch
            else:
                players, coords, labels, match_ids, lengths = batch

            players, coords, labels = players.to(device), coords.to(device), labels.to(device)
            seq_mask = make_seq_mask(lengths).to(device)
            outputs = model(players, coords, src_key_padding_mask=seq_mask)
            if permute is not None:
                labels = permute(labels, outputs, match_ids, seq_mask)

            loss = criterion(outputs.view(-1, model.transformer.num_classes), labels)
            penalty_loss = penalty(outputs) if penalty is not None else 0
            batch_size = float(lengths.sum())
            batch_losses.append(float(loss + penalty_loss))
            batch_sizes.append(batch_size)

            acc_loss += loss
            acc_penalty_loss += penalty_loss

            total_loss = acc_loss + acc_penalty_loss
            acc_num_batches += batch_size

            pbar.set_description(msg.format(batch_idx + 1, len(loader), total_loss / (acc_num_batches - 1)))
            pbar.refresh()

    acc_loss /= acc_num_batches
    if penalty:
        acc_penalty_loss /= acc_num_batches
    total_loss = acc_loss + acc_penalty_loss

    if wandb.run is not None:
        wandb.log({"val/val_CriterionLoss": acc_loss})
        wandb.log({"val/val_PenaltyLoss": acc_penalty_loss})
        wandb.log({"val/val_TotalLoss": total_loss})
    return total_loss, batch_losses, batch_sizes


def train_transformer_single_match(split, conf, args, device):
    model = build(conf).to(device)
    model.backbone.requires_grad_(not conf.optimization.freeze_backbone)

    optimizer = optim.Adam(model.parameters(), lr=conf.optimization.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=conf.data.padding_token)

    if conf.optimization.player_count is not None:
        penalty = SequenceCountPenalty(conf.optimization.player_count.gamma,
                                       conf.optimization.player_count.max_count_per_class,
                                       ignore_index=conf.data.padding_token)
    else:
        penalty = None

    if conf.optimization.scheduler is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=conf.optimization.scheduler.factor,
                                                               patience=conf.optimization.scheduler.patience,
                                                               min_lr=conf.optimization.scheduler.min_lr)
    else:
        scheduler = None

    train_dataset = PlayerPatchesDataset(split.train,
                                         patch_transform=player_transform('transformer'),
                                         coord_transform=RandomCoordsHorizontalFlip(0.5),
                                         num_processes=args.num_loading_threads,
                                         load_gt=conf.data.use_groundtruth)
    train_loader = DataLoader(dataset=train_dataset,
                              collate_fn=CollateFrames(num_data_outputs=train_dataset.num_data_outputs,
                                                       num_classes=conf.transformer.model.num_classes,
                                                       padding_token=conf.data.padding_token),
                              shuffle=True,
                              batch_size=conf.optimization.batch_size,
                              num_workers=args.num_workers)

    valid_dataset = PlayerPatchesDataset(split.valid,
                                         num_processes=args.num_loading_threads,
                                         load_gt=False)
    valid_loader = DataLoader(dataset=valid_dataset,
                              collate_fn=CollateFrames(num_data_outputs=valid_dataset.num_data_outputs,
                                                       num_classes=conf.transformer.model.num_classes,
                                                       padding_token=conf.data.padding_token),
                              shuffle=True,
                              batch_size=conf.optimization.batch_size,
                              num_workers=args.num_workers)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        num_bad_epochs = checkpoint['num_bad_epochs']
    else:
        start_epoch = 0
        best_loss = np.Inf
        num_bad_epochs = 0

    for epoch in range(start_epoch, conf.optimization.num_epochs):
        train(model, train_loader, optimizer, criterion, penalty, None, device, epoch + 1, args.log_freq)

        valid_loss = evaluate(model, valid_loader, criterion, penalty, None, device)[0]
        print(f"Validation loss = {valid_loss}")
        if valid_loss < best_loss:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': valid_loss,
                        'num_bad_epochs': num_bad_epochs},
                       args.weights_dir.joinpath('model.pth'))
            best_loss = valid_loss
            num_bad_epochs = 0
        else:
            num_bad_epochs += 1

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_loss,
                    'num_bad_epochs': num_bad_epochs},
                   args.weights_dir.joinpath('checkpoint.pth'))

        if scheduler:
            scheduler.step(valid_loss)

        if num_bad_epochs >= conf.optimization.early_stopping_patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break


def train_transformer_multiple_matches(conf, args, device):
    model = build(conf).to(device)
    model.backbone.requires_grad_(not conf.optimization.freeze_backbone)

    optimizer = optim.Adam(model.parameters(), lr=conf.optimization.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=conf.data.padding_token)

    if conf.optimization.player_count is not None:
        penalty = SequenceCountPenalty(conf.optimization.player_count.gamma,
                                       conf.optimization.player_count.max_count_per_class,
                                       ignore_index=conf.data.padding_token)
    else:
        penalty = None

    if conf.optimization.scheduler is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=conf.optimization.scheduler.factor,
                                                               patience=conf.optimization.scheduler.patience,
                                                               min_lr=conf.optimization.scheduler.min_lr)
    else:
        scheduler = None

    # Clustered predictions dataset
    cluster_method = 'gk_separation' if conf.clustering.separate_goalkeepers else 'spectral'

    train_dataset = CombineMatchesDataset(args.splits.train,
                                          # patch_transform=player_transform('transformer'),
                                          # coord_transform=RandomCoordsHorizontalFlip(0.5),
                                          num_processes=args.num_loading_threads,
                                          load_predictions=True,
                                          use_background=conf.backbone.background,
                                          cluster_method=cluster_method,
                                          num_frames_per_half=conf.optimization.num_frames_per_half,
                                          similarity_threshold=conf.optimization.similarity_threshold,
                                          num_random_comb_per_step=conf.optimization.num_random_comb_per_step,
                                          mix=conf.optimization.mix,
                                          kit_clusters_csv=args.kit_clusters_csv)

    train_permute = PermutationInvariant(conf.data.num_classes,
                                         ignore_index=conf.data.padding_token,
                                         device=device)
    valid_permute = PermutationInvariant(conf.data.num_classes,
                                         ignore_index=conf.data.padding_token,
                                         device=device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        num_bad_epochs = checkpoint['num_bad_epochs']
    else:
        start_epoch = 0
        best_loss = np.Inf
        num_bad_epochs = 0

    valid_dataset = CombineMatchesDataset(args.splits.train,
                                          num_processes=args.num_loading_threads,
                                          load_predictions=True,
                                          use_background=conf.backbone.background,
                                          cluster_method=cluster_method,
                                          num_frames_per_half=500,
                                          similarity_threshold=conf.optimization.similarity_threshold,
                                          num_random_comb_per_step=conf.optimization.num_random_comb_per_step,
                                          mix=conf.optimization.mix,
                                          kit_clusters_csv=args.kit_clusters_csv)

    # train_dataset.start_epoch()
    # train_dataset.load_step_data()
    # train_loader = DataLoader(dataset=train_dataset,
    #                           collate_fn=CollateFrames(num_data_outputs=train_dataset.num_data_outputs,
    #                                                    num_classes=conf.transformer.model.num_classes,
    #                                                    padding_token=conf.data.padding_token),
    #                           shuffle=True,
    #                           batch_size=conf.optimization.batch_size,
    #                           num_workers=args.num_workers)
    #
    valid_dataset.start_epoch()
    valid_dataset.load_all_data()
    valid_loader = DataLoader(dataset=valid_dataset,
                              collate_fn=CollateFrames(num_data_outputs=valid_dataset.num_data_outputs,
                                                       num_classes=conf.transformer.model.num_classes,
                                                       padding_token=conf.data.padding_token),
                              shuffle=False,
                              batch_size=conf.optimization.batch_size,
                              num_workers=args.num_workers)

    for epoch in range(start_epoch, conf.optimization.num_epochs):
        if wandb.run is not None and scheduler:
            print(f"Learning rate {optimizer.param_groups[0]['lr']}")
            wandb.log({"learning_rate": optimizer.param_groups[0]['lr']})

        train_dataset.start_epoch()
        batch_losses, batch_sizes = [], []
        while train_dataset.load_step_data() > 0:
            train_loader = DataLoader(dataset=train_dataset,
                                      collate_fn=CollateFrames(num_data_outputs=train_dataset.num_data_outputs,
                                                               num_classes=conf.transformer.model.num_classes,
                                                               padding_token=conf.data.padding_token),
                                      shuffle=True,
                                      batch_size=conf.optimization.batch_size,
                                      num_workers=args.num_workers)
            batch_losses_, batch_sizes_ = train(model, train_loader, optimizer, criterion, penalty, train_permute,
                                                device, epoch + 1, args.log_freq)
            batch_losses.extend(batch_losses_)
            batch_sizes.extend(batch_sizes_)
        train_dataset.clear_epoch_data()

        batch_losses, batch_sizes = np.asarray(batch_losses), np.asarray(batch_sizes)
        epoch_loss = batch_losses.dot(batch_sizes) / batch_sizes.sum()

        if wandb.run is not None:
            metrics = {"train/epoch/loss": epoch_loss}
            wandb.log(metrics)

        # valid_dataset.start_epoch()
        batch_losses, batch_sizes = [], []
        # while valid_dataset.load_step_data() > 0:
        #     valid_loader = DataLoader(dataset=valid_dataset,
        #                               collate_fn=CollateFrames(num_data_outputs=valid_dataset.num_data_outputs,
        #                                                        num_classes=conf.transformer.model.num_classes,
        #                                                        padding_token=conf.data.padding_token),
        #                               shuffle=True,
        #                               batch_size=conf.optimization.batch_size,
        #                               num_workers=args.num_workers)
        #     _, batch_losses_, batch_sizes_ = evaluate(model, valid_loader, criterion, penalty, valid_permute, device)
        #     batch_losses.extend(batch_losses_)
        #     batch_sizes.extend(batch_sizes_)
        # valid_dataset.clear_epoch_data()

        _, batch_losses_, batch_sizes_ = evaluate(model, valid_loader, criterion, penalty, valid_permute, device)
        batch_losses.extend(batch_losses_)
        batch_sizes.extend(batch_sizes_)

        batch_losses, batch_sizes = np.asarray(batch_losses), np.asarray(batch_sizes)
        valid_loss = batch_losses.dot(batch_sizes) / batch_sizes.sum()

        print(f"Validation loss = {valid_loss}")
        if wandb.run is not None:
            metrics = {"valid/loss": valid_loss}
            wandb.log(metrics)

        if valid_loss < best_loss:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': valid_loss,
                        'num_bad_epochs': num_bad_epochs},
                       args.weights_dir.joinpath('model.pth'))
            best_loss = valid_loss
            num_bad_epochs = 0
        else:
            num_bad_epochs += 1

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_loss,
                    'num_bad_epochs': num_bad_epochs},
                   args.weights_dir.joinpath('checkpoint.pth'))

        if scheduler:
            scheduler.step(valid_loss)

        if num_bad_epochs >= conf.optimization.early_stopping_patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break


def powspace(start: float, stop: float, num: int):
    log_start, log_stop = np.log(start), np.log(stop)
    return np.exp(np.linspace(log_start, log_stop, num))


def find_lr(conf, args, device):
    model = build(conf).to(device)
    model.backbone.requires_grad_(not conf.optimization.freeze_backbone)

    criterion = nn.CrossEntropyLoss(ignore_index=conf.data.padding_token)

    if conf.optimization.player_count is not None:
        penalty = SequenceCountPenalty(conf.optimization.player_count.gamma,
                                       conf.optimization.player_count.max_count_per_class,
                                       ignore_index=conf.data.padding_token)
    else:
        penalty = None

    # Clustered predictions dataset
    cluster_method = 'gk_separation' if conf.clustering.separate_goalkeepers else 'spectral'

    train_dataset = CombineMatchesDataset(args.splits.train,
                                          # patch_transform=player_transform('transformer'),
                                          # coord_transform=RandomCoordsHorizontalFlip(0.5),
                                          num_processes=args.num_loading_threads,
                                          load_predictions=True,
                                          use_background=conf.backbone.background,
                                          cluster_method=cluster_method,
                                          num_frames_per_half=conf.optimization.num_frames_per_half,
                                          similarity_threshold=conf.optimization.similarity_threshold,
                                          num_random_comb_per_step=conf.optimization.num_random_comb_per_step,
                                          mix=conf.optimization.mix,
                                          kit_clusters_csv=args.kit_clusters_csv)

    train_permute = PermutationInvariant(conf.data.num_classes,
                                         ignore_index=conf.data.padding_token,
                                         device=device)

    model.train()

    train_dataset.start_epoch()
    msg = 'Epoch {} Iteration {}: Loss = {}'

    if train_dataset.load_step_data() == 0:
        return

    train_loader = DataLoader(dataset=train_dataset,
                              collate_fn=CollateFrames(num_data_outputs=train_dataset.num_data_outputs,
                                                       num_classes=conf.transformer.model.num_classes,
                                                       padding_token=conf.data.padding_token),
                              shuffle=True,
                              batch_size=conf.optimization.batch_size,
                              num_workers=args.num_workers)

    num_batches = min(500, len(train_loader))

    learning_rates = powspace(1e-10, 2, num_batches)
    batch_losses, batch_sizes = [], []
    for batch_idx, batch in (batches_bar := tqdm(enumerate(train_loader))):
        if batch_idx == num_batches:
            break
        optimizer = optim.Adam(model.parameters(), lr=learning_rates[batch_idx])

        if train_loader.dataset.num_data_outputs == 3:
            players, coords, labels, lengths = batch
        else:
            players, coords, labels, match_ids, lengths = batch

        players, coords, labels = players.to(device), coords.to(device), labels.to(device)
        seq_mask = make_seq_mask(lengths).to(device)

        optimizer.zero_grad()
        outputs = model(players, coords, src_key_padding_mask=seq_mask)

        if train_permute is not None:
            labels = train_permute(labels, outputs, match_ids, seq_mask)

        criterion_loss = criterion(outputs.view(-1, model.transformer.num_classes), labels)
        penalty_loss = penalty(outputs, seq_mask) if penalty is not None else 0

        loss = criterion_loss + penalty_loss
        loss.backward()
        optimizer.step()

        batch_losses.append(float(loss))
        batch_sizes.append(float(lengths.sum()))

        batch_losses_, batch_sizes_ = np.asarray(batch_losses), np.asarray(batch_sizes)
        cumulative_loss = batch_losses_.dot(batch_sizes_) / batch_sizes_.sum()

        metrics = {"train/batch/epoch": 1,
                   "train/batch/iteration": batch_idx,
                   "train/batch/criterion_loss": criterion_loss,
                   "train/batch/penalty_loss": penalty_loss,
                   "train/batch/train_loss": loss,
                   "train/batch/cumulative_loss": cumulative_loss,
                   "train/batch/learning_rate": learning_rates[batch_idx]}

        batches_bar.set_description(msg.format(1, batch_idx + 1, loss))
        batches_bar.refresh()
        if wandb.run is not None:
            wandb.log(metrics)
