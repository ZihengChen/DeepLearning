from pylab import *
from torch.utils.data import Dataset, DataLoader


class tcDataset(Dataset):
    def __init__(self, arr, shape=None , transform=None ):
        self.label = arr[:,-1]  .astype('int')
        self.data  = arr[:,0:-1].astype('float32')

        if not shape is None:
            self.data  = np.reshape(self.data, shape)

        self.transform = transform
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'label': self.label[idx]}
        if self.transform:
            sample['data'] = self.transform(sample['data'])
        return sample
