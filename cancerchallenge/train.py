import torch
import logging
from cancerchallenge.model import Model
from cancerchallenge.data_io import CancerChallengeDataset
from torch.utils.data import DataLoader
from cancerchallenge.loss import CrossEntropyLoss
from torch.optim import Adam


# Train function
def train(inputdir_train, inputlabels_train):
    model = Model()

    dataset = CancerChallengeDataset(inputdir_train, inputlabels_train)

    dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=True)

    optim = Adam(model.parameters(), lr=1e-3)

    criterion = CrossEntropyLoss()

    for epoch in range(5):
        for batch_no, (images, labels) in enumerate(dataloader):
            model_output = model(images)

            loss = criterion(model_output, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()





