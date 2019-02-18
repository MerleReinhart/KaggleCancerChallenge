from torch.utils.data import Dataset
import torch
import pandas as pd
import os
from skimage import io
from torchvision.transforms import ToTensor


class CancerChallengeDataset(Dataset):

    def __init__(self, inputdir, inputlabels):
        self.image_id = pd.read_csv(inputlabels)
        self.dir = inputdir

    def __getitem__(self, idx):
        image_name = os.path.join(self.dir, self.image_id.iloc[idx, 0]+'.tif')
        image = io.imread(image_name)
        image = ToTensor()(image)

        label = self.image_id.iloc[idx, 1]
        label = torch.tensor(label,dtype=torch.float)
        label = label.reshape(1)

        return image, label

    def __len__(self):
        return len(self.image_id)
