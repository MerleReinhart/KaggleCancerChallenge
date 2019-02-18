import torch
from cancerchallenge.data_io import CancerChallengeDataset
from cancerchallenge.model import Model
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def predict(model_filename, image_dir, image_labels):
    model = Model()
    model.load_state_dict(torch.load(model_filename))

    dataset = CancerChallengeDataset(image_dir, image_labels)
    n_testimages = 20000
    validation_indices = list(range(n_testimages))
    validation_sampler = SubsetRandomSampler(validation_indices)
    dataloader = DataLoader(dataset=dataset, sampler=validation_sampler, batch_size=100)

    n_correct_labels = 0

    for image_no, (images, labels) in enumerate(dataloader):
        output = model(images)
        output = output.detach().numpy()
        labels = labels.detach().numpy()

        threshold = 0.5
        predicted_label = (output > threshold)
        true_labels = (labels > threshold)
        correct_labels = predicted_label==true_labels
        n_correct_labels = n_correct_labels + np.sum(correct_labels)

    accuracy = n_correct_labels/n_testimages

    print('Percentage of true predictions = ', accuracy)
    return accuracy


predict('/Users/mreinhar/A_PythonML/Kaggle/CancerChallenge/models/mymodel.pt',
        '/Users/mreinhar/A_PythonML/Kaggle/CancerChallenge/data/train/',
        '/Users/mreinhar/A_PythonML/Kaggle/CancerChallenge/data/train_labels.csv')
