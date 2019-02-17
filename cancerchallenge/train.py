import torch
import logging
from cancerchallenge.model import Model
from cancerchallenge.data_io import CancerChallengeDataset

# Train function
def train(inputdir_train, inputlabels_train):
    model = Model()

    dataset = CancerChallengeDataset(inputdir_train, inputlabels_train)


