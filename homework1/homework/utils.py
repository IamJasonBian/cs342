from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os


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
            #organized labels as a list of lists and skip the header
            self.labels = [line.split(',') for line in f.readlines()][1:]
            

    def __len__(self):
        """
        Your code here
        """
        return len(self.labels) - 1

def __getitem__(self, idx):
        image_name = self.image_paths[idx]
        image_path = os.path.join(self.dataset_path, image_name)
        label = int(image_name.split('.')[0])  # Extract label from the image name

        image = Image.open(image_path)
        # You can perform any necessary transformations on the image here (e.g., resizing, normalization)

        return image, label

def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()

'''
#run the SuperTuxDataset(Dataset): class
if __name__ == '__main__':
    test = SuperTuxDataset('homework1/data/train')
    #Run getitem
    test.__getitem__(0)
    #Run len
    test.__len__()
'''



