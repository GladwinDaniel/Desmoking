from .base_dataset import BaseDataset, get_transform
from PIL import Image
import os

class SingleDataset(BaseDataset):
    def initialize(self, opt):
        self.root = opt.dataroot
        self.image_paths = sorted([os.path.join(self.root, f) for f in os.listdir(self.root)])
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return {'A': img, 'A_paths': path}

    def __len__(self):
        return len(self.image_paths)

