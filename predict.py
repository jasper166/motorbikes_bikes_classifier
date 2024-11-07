import numpy as np 
import cv2
import os
import torch
import torchvision
import argparse

from torch import nn
from torchvision import transforms
from config import config

from utils import utils_save_cfg, utils_model, utils_loss
import random

class Model():
    
    def __init__(self, load_height, load_width ,
                 checkpoint_dir, device):

        # self.model = torchvision.models.resnet101(pretrained = False)
        # self.model = utils_model.create_model(name_model= 'resnet50', num_classes= 2)
        # num_features = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_features, 2) # khởi tạo mô hình = trẻ 3 tuổi

        self.model = torchvision.models.vgg16()
        # num_features = self.model.classifier[6].in_features
        # self.model.classifier[6] = nn.Linear(num_features, 2)
        num_classes = 2  # Số lượng lớp đầu ra (ví dụ: xe máy và xe đạp)
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # Lớp Fully Connected thứ nhất
            nn.LeakyReLU(),
            nn.Dropout(0.5),  # Thêm Dropout với xác suất bị bỏ (dropout probability) là 0.5
            nn.Linear(4096, 2048),  # Lớp Fully Connected thứ hai
            nn.LeakyReLU(),  
            nn.Dropout(0.5),  # Thêm Dropout với xác suất bị bỏ là 0.5
            nn.Linear(2048, num_classes)  # Lớp đầu ra cho số lượng lớp của bạn
        )
        self.device = device # gán biến
        self.model.to(self.device)

        self.checkpoint_model = checkpoint_dir
 
        self.model.load_state_dict(torch.load(self.checkpoint_model, map_location=torch.device(self.device))['model_state_dict']) # load mô hình đã cho huấn luyện # 10 tuổi

        self.load_height = load_height
        self.load_width = load_width
        self.labels = {0:'xedap', 1: 'xemay'}

        self.transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((self.load_height, self.load_width)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # hàm chỉnh sửa ảnh về dạng khác

        self.model.eval() # chuyển mô hình về dự đoán 
        print("Start ...")
        

    def preprocess(self, path_image):

        img = cv2.imread(path_image) # đọc ảnh
        img  = self.transform(img) # chuyển ảnh về dangh tensor
        return img.to(self.device).unsqueeze(0) # thêm 1 chiều cho ảnh 1,n,h,w thì phải # 1*3*224*224

    def predict(self, path_image):
        input = self.preprocess(path_image) # = 1 ảnh
        # self.model.eval()
        with torch.no_grad(): # tắt đạo hàm
            output = self.model(input) # dự đoán ảnh # [0.3 0.7] [[0.6 0.4]] = [0.6 0.4]
            output = output.softmax(1).to('cpu').numpy() # chuyển kết quả về numpy
            
        score = np.mean(output, axis=0)
        label = np.argmax(score) # lấy vị trí có xác suất cao nhất P(xj|x1x2...)
        return self.labels[label], score[label] # Dog, 0.7
    
def main(args):
    # khởi tạo các biến
    cfg = config[args.config]

    CHECKPOINT_DIR = args.checkpoint_dir
    #model

    DEVICE = cfg['DEVICE']

    #data
    RESIZE = cfg['RESIZE']
    LOAD_WIDTH = cfg['LOAD_WIDTH']
    LOAD_HEIGHT = cfg['LOAD_HEIGHT']
    IMAGE = args.image
    
    dict_class = {'xedap': 0, 'xemay' : 1}
    folder_xedap = os.path.join(args.dir_root, 'test/xedap')
    folder_xemay = os.path.join(args.dir_root, 'test/xemay')

    # Lấy danh sách các tệp tin trong thư mục xedap và xemay
    images_xedap = [os.path.join(folder_xedap, filename) for filename in os.listdir(folder_xedap) if filename.endswith('.jpg')]
    images_xemay = [os.path.join(folder_xemay, filename) for filename in os.listdir(folder_xemay) if filename.endswith('.jpg')]

    # Kết hợp danh sách ảnh từ hai thư mục
    all_images = images_xedap + images_xemay
    # print(all_images)
    # Sắp xếp ngẫu nhiên danh sách ảnh
    # random.shuffle(all_images)
    # print(all_images)
    model = Model(load_height= LOAD_HEIGHT, load_width= LOAD_WIDTH,\
                  checkpoint_dir= CHECKPOINT_DIR, device= DEVICE)
    gt_labels = []
    pre_labels = []

    for i, image in enumerate(all_images):
        label, score = model.predict(image)
        pre_labels.append(dict_class[label])
        gt_label = dict_class.get(os.path.basename(os.path.dirname(image)), -1)
        gt_labels.append(gt_label)
        # print(image, label)
    # print('path_image: {} \nlabel : {} \nxac xuat: {}'.format(IMAGE, label, score))

    # Tổng số mẫu
    total_samples = len(gt_labels)

    # Số lần dự đoán đúng
    correct_predictions = sum(1 for true_label, pred_label in zip(gt_labels, pre_labels) if true_label == pred_label)

    # Độ chính xác
    accuracy = correct_predictions / total_samples

    # Độ chính xác sẽ là một số từ 0 đến 1, hoặc có thể nhân 100 để biểu diễn dưới dạng phần trăm.
    accuracy_percentage = accuracy * 100

    print(f"Độ chính xác của mô hình là: {accuracy_percentage:.2f}%")
    
def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type= str, default= 'test')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--image', type= str, default= 'image.jpg')
    parser.add_argument('--dir_root', type= str, default= '/root')

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    args = get_args_parser()
    main(args=args)
