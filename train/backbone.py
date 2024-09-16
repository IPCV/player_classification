import torch
import torch.optim as optim
import wandb
from pytorch_metric_learning import distances
from pytorch_metric_learning import losses
from pytorch_metric_learning import miners
from pytorch_metric_learning import reducers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm


from models.backbone import Backbone
from util.data import ImageFolders, \
    RandomSeparateFolderSampler
from util.misc import get_all_embeddings
from util.transforms import player_transform


__all__ = ['train_backbone']


def train(model, loader, optimizer, loss_func, mining_func, device, epoch, log_freq=20):
    if isinstance(mining_func, miners.TripletMarginMiner):
        msg = 'Epoch {} Iteration {}: Loss = {}, Num mined triplets = {}'
    else:
        msg = 'Epoch {} Iteration {}: Loss = {}, Num +mined triplets = {}, Num -mined triplets = {}'

    model.train()
    for batch_idx, (data, labels) in (batches_bar := tqdm(enumerate(loader))):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()

        metrics = {"train/train_loss": loss,
                   "train/epoch": epoch,
                   "train/iteration": batch_idx}

        if not hasattr(mining_func, 'num_triplets'):
            metrics["train/num_neg_pairs"] = mining_func.num_neg_pairs
            metrics["train/num_pos_pairs"] = mining_func.num_pos_pairs
            metrics["train/mined_triplets"] = mining_func.num_pos_pairs + mining_func.num_neg_pairs
            batches_bar.set_description(
                msg.format(epoch, batch_idx, loss, mining_func.num_pos_pairs, mining_func.num_neg_pairs))
        else:
            metrics["train/mined_triplets"] = mining_func.num_triplets
            batches_bar.set_description(msg.format(epoch, batch_idx, loss, mining_func.num_triplets))

        batches_bar.refresh()
        if (batch_idx + 1) % log_freq == 0:
            wandb.log(metrics)



def evaluate(model, train_sets, val_sets, accuracy_calculator):
    msg = 'Validation Matches {}/{}: MAP@R = {}'

    num_correct_samples, num_test_samples = 0, 0
    for i, (train_set, val_set) in (pbar := tqdm(enumerate(zip(train_sets, val_sets)))):
        train_embeddings, train_labels = get_all_embeddings(train_set, model)
        val_embeddings, val_labels = get_all_embeddings(val_set, model)
        train_labels = train_labels.squeeze(1)
        val_labels = val_labels.squeeze(1)
        accuracies = accuracy_calculator.get_accuracy(
            val_embeddings, val_labels, train_embeddings, train_labels, False
        )
        num_correct_samples += accuracies['mean_average_precision_at_r'] * len(val_labels)
        num_test_samples += len(val_labels)

        pbar.set_description(msg.format(i + 1, len(val_sets), num_correct_samples / num_test_samples))
        pbar.refresh()

    mapar = num_correct_samples / num_test_samples
    wandb.log({"val/val_MAP@R": mapar})
    print(f"Validation set accuracy (MAP@R) = {mapar}")
    return mapar


def train_backbone(conf, args, device):
    model = Backbone(conf.backbone.model.name, conf.backbone.model.dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=conf.optimization.learning_rate)

    if conf.backbone.metric.distance == 'LP':
        distance = distances.LpDistance()
    else:
        distance = distances.CosineSimilarity()

    if conf.backbone.metric.loss == 'NTXentLoss' or conf.backbone.metric.loss == 'InfoNCE':
        reducer = reducers.MeanReducer()
        loss_func = losses.NTXentLoss(temperature=conf.backbone.metric.temperature, distance=distance, reducer=reducer)
    else:
        reducer = reducers.ThresholdReducer(low=0)
        loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)

    if conf.backbone.metric.miner == 'TripletMarginMiner':
        mining_func = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="semihard")
    else:
        mining_func = miners.BatchEasyHardMiner(pos_strategy=miners.BatchEasyHardMiner.EASY,
                                                neg_strategy=miners.BatchEasyHardMiner.SEMIHARD,
                                                allowed_pos_range=None,
                                                allowed_neg_range=None,
                                                distance=distance)

    accuracy_calculator = AccuracyCalculator(include=("mean_average_precision_at_r",), k='max_bin_count')

    data_dir = 'data_bkg' if conf.backbone.background else 'data'
    match_paths = [m.joinpath('player_labeling', data_dir) for m in args.matches]
    train_sets, val_sets, train_dirs = [], [], []
    for m in match_paths:
        train_dir, val_dir = m.joinpath('train'), m.joinpath('valid')
        train_sets.append(ImageFolder(root=train_dir, transform=player_transform('backbone')))
        val_sets.append(ImageFolder(root=val_dir, transform=player_transform()))
        train_dirs.append(train_dir)
    image_folders = ImageFolders(train_dirs, transform=player_transform('backbone'))

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_mapar = checkpoint['map@r']
    else:
        start_epoch = 0
        best_mapar = 0

    for epoch in range(start_epoch, conf.optimization.num_epochs):
        batch_sampler = RandomSeparateFolderSampler(image_folders,
                                                    batch_size=conf.optimization.batch_size,
                                                    drop_last=False)
        train_loader = DataLoader(dataset=image_folders,
                                  batch_sampler=batch_sampler,
                                  num_workers=args.num_workers)
        
        train(model, train_loader, optimizer, loss_func, mining_func, device, epoch + 1, args.log_freq)
        mapar = evaluate(model, train_sets, val_sets, accuracy_calculator)

        if mapar > best_mapar:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'map@r': best_mapar},
                       args.weights_dir.joinpath('model.pth'))
            best_mapar = mapar

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'map@r': best_mapar},
                   args.weights_dir.joinpath('checkpoint.pth'))
