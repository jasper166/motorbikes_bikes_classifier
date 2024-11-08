import cv2
import os
import torch # 
import numpy as np 
from torch.utils import data
from torchvision import transforms

class Datasets(data.Dataset) : 
    def __init__(self, path_data, load_width, load_height, nb_classes) -> None:
        super(Datasets, self).__init__()
        self.load_width = load_width
        self.load_height = load_height # 24*224*3
        self.nb_classes = nb_classes # numbers classes = 2 
        self.path_data = path_data
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.load_height, self.load_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #RBG - RGB 0-1 tensor type numpy int float gpu để train
            transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomRotation(degrees=30),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        ])
        path_images = [] # đường dẫn  data/train/xemay/xemay_0.jpeg
        labels = [] # 0 1 0 1 0 1 0 1 1 1 1  0 0 0 1 0 1 0 1 0 0 
        dict_class = {'xedap': 0, 'xemay' : 1}
        
        for label in os.listdir(self.path_data) : 
            for fname in os.listdir(os.path.join(self.path_data, label)) :
                if '.jpeg' in fname or '.png' in fname :
                    path_images.append(os.path.join(self.path_data, label, fname))
                    labels.append(dict_class[label]) 
        
        self.path_images = path_images
        self.labels = labels

    def __getitem__(self, index) :
        path_image = self.path_images[index]
        label = self.labels[index]
        image = cv2.imread(path_image) # đọc ảnh return 480*360*3 0-255
        image = self.transform(image) # 0-1 
        result = {
            'path_image' : path_image,
            'image' : image,
            'label' : label
        }
        return result
    
    def __len__(self):
        return len(self.labels)

    