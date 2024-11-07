import cv2
import os
import torch
import numpy as np 
from torch.utils import data
from torchvision import transforms
class DatasetDogCat(data.DataLoader) :
    def __init__(self, opt):
        super().__init__()
        super(DatasetDogCat, self).__init__()
        self.load_width = opt.load_width
        self.load_height = opt.load_height
        self.phare = opt.phare
        self.nb_classes = opt.nb_classes
        self.root = opt.root
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        path_images = []
        labels = []
        for label in os.listdir(self.root) :
            for fname in os.listdir(os.path.join(self.root, label)) :
                path_images.append(os.path.join(self.root, label, fname))
                labels.append(label)
        
        self.path_images = path_images
        self.labels = labels
    def __getitem__(self, index) :
        path_image = self.path_images[index]
        image = cv2.imread(path_image)
        image = transforms(image)
        result = {
            'path_image' : path_image,
            'image' : image
        }
        return result
    
    def __len__(self) -> int:
        return len(self.path_images)

    