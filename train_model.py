# Copyright (c) 2025 Brigham Skarda

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.sparse import spmatrix
from collections.abc import Iterable
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from scipy.sparse import spmatrix
from data_cleaner import NUM_AUTHORS

DEVICE = torch.device("cuda")


class EmailDataSet(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.__x: spmatrix = x
        self.__y = y

    def __getitem__(self, index):
        return (
            torch.Tensor(self.__x.getrow(index).todense()).squeeze().to(DEVICE),
            self.__y[index],
        )

    def __len__(self):
        return len(self.__y)


def main():
    print("Reading email data")
    train_data = pd.read_csv("./data/train_emails.csv")
    test_data = pd.read_csv("./data/test_emails.csv")

    print("Tokenizing emails")
    tfidf = TfidfVectorizer(
        tokenizer=word_tokenize,
        preprocessor=None,
        token_pattern=None,
        max_features=25000,
    )
    tokenized_train_emails = tfidf.fit_transform(train_data["content"])
    tokenized_test_emails = tfidf.transform(test_data["content"])
    print("Tokenized train data shape:", tokenized_train_emails.get_shape())
    num_features = tokenized_train_emails.get_shape()[1]

    HIDDEN_LAYER_SIZE = 1000
    BATCH_SIZE = 32
    N_ITERATIONS = 2940 * 3
    model = nn.Sequential(
        nn.Linear(num_features, HIDDEN_LAYER_SIZE),
        nn.BatchNorm1d(HIDDEN_LAYER_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
        nn.BatchNorm1d(HIDDEN_LAYER_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
        nn.BatchNorm1d(HIDDEN_LAYER_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
        nn.BatchNorm1d(HIDDEN_LAYER_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
        nn.BatchNorm1d(HIDDEN_LAYER_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_LAYER_SIZE, NUM_AUTHORS),
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters())

    train_dataset = EmailDataSet(
        tokenized_train_emails,
        torch.Tensor(
            pd.get_dummies(train_data["author"]).to_numpy(),
        ).to(DEVICE),
    )
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = EmailDataSet(
        tokenized_test_emails,
        torch.Tensor(
            pd.get_dummies(test_data["author"]).to_numpy(),
        ).to(DEVICE),
    )
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_losses, test_losses = train(
        model, optimizer, train_dataloader, test_dataloader, N_ITERATIONS
    )
    plot_losses(train_losses, test_losses)
    print(f"Final train loss: {train_losses[-1]}")
    print(f"Final test loss: {test_losses[-1]}")
    print(f"Final train accuracy: {evaluate_model(model, train_dataloader)}")
    print(f"Final test accuracy: {evaluate_model(model, test_dataloader)}")


def plot_losses(train_losses: list[float], test_losses: list[float]):
    if not os.path.exists("./imgs"):
        os.makedirs("./imgs")
    x_axis = [a * 25 for a, _ in enumerate(train_losses)]
    plt.plot(x_axis, train_losses)
    plt.title("Train Losses")
    plt.xlabel("Batch Number")
    plt.ylabel("Loss")
    plt.savefig("./imgs/train.png")
    plt.figure()
    plt.plot(x_axis, test_losses)
    plt.title("Test Losses")
    plt.xlabel("Batch Number")
    plt.ylabel("Loss")
    plt.savefig("./imgs/test.png")


def evaluate_model(model: nn.Module, dataloader: DataLoader) -> float:
    """Returns the accuracy of the model over the whole dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            true_labels = torch.argmax(labels, dim=1)
            total += labels.size(0)
            correct += (predicted == true_labels).sum().item()
    accuracy = correct / total if total > 0 else 0
    return accuracy


def get_tf_idf_encodings(emails: Iterable[str]) -> spmatrix:
    print("Fitting tokenizer")
    tfidf = TfidfVectorizer(
        tokenizer=word_tokenize, preprocessor=None, token_pattern=None
    )
    tfidf.fit(emails)

    print("Getting email tokenizations")
    return tfidf.transform(emails)


def train(
    model: torch.nn,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    n_batches: int,
    loss_interval: int = 25,
) -> tuple[list[float], list[float]]:
    """Returns a list of train losses, and a list of test losses."""
    start_time = time.time()
    cel = nn.CrossEntropyLoss()
    train_losses = []
    test_losses = []
    for i in range(n_batches):
        optimizer.zero_grad()
        emails, tgt = next(iter(train_dataloader))
        output = model(emails)
        loss = cel(output, tgt)
        loss.backward()
        optimizer.step()
        if i % loss_interval == 0:
            print(f"batch {i}/{n_batches}")
            train_losses.append(loss.item())
            with torch.no_grad():
                emails, tgt = next(iter(test_dataloader))
                output = model(emails)
                loss = cel(output, tgt)
                test_losses.append(loss.item())

    end_time = time.time()
    elapsed_time = end_time - start_time
    time_per_batch = elapsed_time / n_batches
    print(f"Average time per batch {time_per_batch:.5f} seconds")
    print(f"Total time training {elapsed_time:.5f} seconds")
    return train_losses, test_losses


if __name__ == "__main__":
    main()
