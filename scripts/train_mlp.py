
import argparse
import os

import torch
import torch.nn as nn
import wandb
import glob

from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from digit_depth.train import MLP, Color2NormalDataset
from digit_depth.handlers import get_save_path, find_recent_model
seed = 42
torch.seed = seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = Path(__file__).parent.parent.resolve()


def train(train_loader, epochs, lr):
    model = MLP().to(device)
    wandb.init(project="MLP", name="Color 2 Normal model train")
    wandb.watch(model, log_freq=100)

    model.train()

    learning_rate = lr
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = epochs
    avg_loss=0.0
    loss_record=[]
    cnt=0
    total_step = len(train_loader)
    for epoch in tqdm(range(1, 1 + num_epochs)):
        for i, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)
            avg_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt+=1

            if (i + 1) % 1 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                loss_record.append(loss.item())
                wandb.log({"Mini-batch loss": loss})
        wandb.log({'Running loss': avg_loss / cnt})
    os.makedirs(f"{base_path}/models", exist_ok=True)
    print(f"Saving model to {base_path}/models/")
    save_name = get_save_path(seed, head=f"{base_path}/models/")
    torch.save(model,f"{save_name}.ckpt")


def test(test_loader,criterion):
    most_recent_model = find_recent_model(f"{base_path}/models")
    model = torch.load(most_recent_model).to(device)
    model.eval()
    wandb.init(project="MLP", name="Color 2 Normal model test")
    wandb.watch(model, log_freq=100)
    model.eval()
    avg_loss = 0.0
    cnt = 0
    with torch.no_grad():
        for idx, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            avg_loss += loss.item()
            cnt=cnt+1
            wandb.log({"Mini-batch test loss": loss})
        avg_loss = avg_loss / cnt
        print("Average test loss: {:.4f}".format(avg_loss))
        wandb.log({'Average test loss': avg_loss})


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode', type=str, default='train', help='train or test')
    argparser.add_argument('--batch_size', type=int, default=1600, help='batch size')
    argparser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    argparser.add_argument('--epochs', type=int, default=40, help='epochs')
    argparser.add_argument('--train_path', type=str, default=f'{base_path}/datasets/train_test_split/train.csv',
                           help='data path')
    argparser.add_argument('--test_path', type=str, default=f'{base_path}/datasets/train_test_split/test.csv',
                           help='test data path')
    option = argparser.parse_args()

    if option.mode == "train":
        train_set = Color2NormalDataset(
            option.train_path)
        train_loader = DataLoader(train_set, batch_size=option.batch_size, shuffle=True)
        print("Training set size: ", len(train_set))
        train(train_loader, option.epochs,option.learning_rate)
    elif option.mode == "test":
        test_set = Color2NormalDataset(
            option.test_path)
        test_loader = DataLoader(test_set, batch_size=option.batch_size, shuffle=True)
        criterion = nn.MSELoss()
        test(test_loader, criterion)


if __name__ == "__main__":
    main()
