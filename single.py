import os
import argparse
import time
import copy

import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default='/home/mcv/datasets/sunrgbd_lite')
    parser.add_argument('--modality', type=str, choices=['rgb', 'hha'], default='rgb')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=4)
    return parser.parse_args()


def build_model(num_classes):
    model = models.alexnet(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    return model


def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs, use_gpu):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        train_loss, train_acc = train_epoch(dataloaders['train'], model, criterion, optimizer, use_gpu)
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(train_loss, train_acc))

        val_loss, val_acc = evaluate_model(dataloaders['val'], model, criterion, use_gpu)
        print('Val Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))

        print()

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        if scheduler is not None:
            scheduler.step(val_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


def train_epoch(dataloader, model, criterion, optimizer, use_gpu):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for data in dataloader:
        inputs, labels = data
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # backward + optimize
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).item()

    loss = running_loss / len(dataloader.dataset)
    acc = running_corrects / len(dataloader.dataset)

    return loss, acc


def evaluate_model(dataloader, model, criterion, use_gpu):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    for data in dataloader:
        inputs, labels = data
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).item()

    loss = running_loss / len(dataloader.dataset)
    acc = running_corrects / len(dataloader.dataset)

    return loss, acc


def main():
    args = parse_args()

    use_gpu = torch.cuda.is_available()

    # data augmentation and normalization for training
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # prepare dataset and dataloaders
    partitions = ['train', 'val', 'test']
    image_datasets = {x: datasets.ImageFolder(os.path.join(args.dataset_dir, x, args.modality), data_transforms[x])
                      for x in partitions}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=(x == 'train'), num_workers=4)
                   for x in partitions}
    print(image_datasets)

    # instantiate the model
    model = build_model(num_classes=len(image_datasets['train'].classes))
    print(model)
    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    scheduler = None

    # train
    model = train_model(dataloaders, model, criterion, optimizer, scheduler, args.epochs, use_gpu)

    # evaluate
    _, train_acc = evaluate_model(dataloaders['train'], model, criterion, use_gpu)
    _, val_acc = evaluate_model(dataloaders['val'], model, criterion, use_gpu)
    _, test_acc = evaluate_model(dataloaders['test'], model, criterion, use_gpu)
    print('Accuracy. Train: {:1.2f}% Val: {:1.2f}% Test: {:1.2f}%'.format(
        train_acc * 100, val_acc * 100, test_acc * 100))


if __name__ == '__main__':
    main()
