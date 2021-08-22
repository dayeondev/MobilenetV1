from PIL import Image
from torch.utils.data.dataset import Dataset
import torch

class FlowersDataset(Dataset):
    def __init__(self, data_path_list, classes, transform=None):
        self.path_list = data_path_list
        self.label = self.get_label(data_path_list)
        self.transform = transform
        self.classes = classes

    def get_label(self, data_path_list):
        label_list = []
        for path in data_path_list:
            label_list.append(path.split('\\')[-2])
        return label_list
    
    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # image = io.imread(self.path_list[idx])
        image = Image.open(self.path_list[idx])

        if self.transform is not None:
            image = self.transform(image)
        return image, self.classes.index(self.label[idx])