from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv

        WARNING: Do not perform data normalization here. 
        """
        self.dataset_path = dataset_path
        with open(self.dataset_path + '/labels.csv', 'r') as f:
            self.labels = f.readlines()

    def __len__(self):
        """
        Your code here
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """

        #split the label into img_path and label (img_path is local path)
        img_path, label = self.labels[idx].split(',')
        
        img = Image.open(self.dataset_path + '/' + img_path)
        img = transforms.ToTensor()(img)
        label = int(label)
        return img, label


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
