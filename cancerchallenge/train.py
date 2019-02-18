import torch
import logging
from cancerchallenge.model import Model
from cancerchallenge.data_io import CancerChallengeDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from cancerchallenge.loss import Loss
from torch.optim import Adam

logger = logging.getLogger(__name__)


# Train function
def train(inputdir_train, inputlabels_train):
    model_filename = '/Users/mreinhar/A_PythonML/Kaggle/CancerChallenge/models/mymodel.pt'
    model = Model()

    dataset = CancerChallengeDataset(inputdir_train, inputlabels_train)

    dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=True)

    optim = Adam(model.parameters(), lr=1e-3)

    criterion = Loss()

    for epoch in range(5):
        total_loss = 0.
        for batch_no, (images, labels) in enumerate(dataloader):
            model_output = model(images)

            loss = criterion(model_output, labels)
            total_loss += loss.item()

            optim.zero_grad()
            loss.backward()
            optim.step()

            logger.debug('Epoch: {}, Batch: {}, Batch loss: {}'.format(epoch, batch_no, loss.item()))

        logger.info('Epoch: {}, Total loss: {}'.format(epoch, total_loss))

    torch.save(model.state_dict(), model_filename)


if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    train('/Users/mreinhar/A_PythonML/Kaggle/CancerChallenge/data/train/',
          '/Users/mreinhar/A_PythonML/Kaggle/CancerChallenge/data/train_labels.csv')

