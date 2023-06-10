from PIL import Image
import csv
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.image_paths = []
        self.labels = []
        
        # Read labels from the CSV file
        with open(os.path.join(dataset_path, 'labels.csv'), 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip the header
            for row in csv_reader:
                image_path = os.path.join(dataset_path, row[0])

                # Convert label string to integer by matching to the self.labels object
                label = LABEL_NAMES.index(row[1])
                self.image_paths.append(image_path)
                self.labels.append(label)
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
            
            #,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()


#run the SuperTuxDataset(Dataset): class
if __name__ == '__main__':
    test = SuperTuxDataset('homework1/data/train')
    #Run getitem
    test.__getitem__(0)
    #Run len
    test.__len__()




