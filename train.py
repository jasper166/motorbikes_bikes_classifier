
import argparse
import datetime
import time
import torch
import os
import copy
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.optim import lr_scheduler

from utils.utils import write_txt
from utils import utils_save_cfg, utils_model, utils_loss
from config import config
from datasets.datasets_train import Datasets
from logs.logs import setup_logger
from engine import train_one_epoch
import torchvision

logger = setup_logger("train.log")
dir_name = os.path.dirname(os.path.realpath(__file__))



def train(args):
    cfg = config[args.config]

    # khởi tạo các biến
    BATCH_SIZE = cfg['BATCH_SIZE']
    EPOCHS = cfg['EPOCHS']
    TRAIN_ON = cfg['TRAIN_ON']
    DATA_ROOT = args.data_root
    CHECKPOINT_DIR = args.checkpoint_dir
    #model
    NAME_MODEL = args.name_model
    NUM_CLASSES = cfg['NUM_CLASSES']
    LR = cfg['LR']
    NUM_WORKERS = cfg['NUM_WORKERS']
    NAME_LOSS = cfg['NAME_LOSS']
    DEVICE = cfg['DEVICE']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']

    #ckpt
    NUM_SAVE_CKPT = cfg['NUM_SAVE_CKPT']
    SAVE_CKPT = cfg['SAVE_CKPT']
    #data
    RESIZE = cfg['RESIZE']
    LOAD_WIDTH = cfg['LOAD_WIDTH']
    LOAD_HEIGHT = cfg['LOAD_HEIGHT']
    # checkpoints/resnet101
    os.makedirs(os.path.join(CHECKPOINT_DIR, NAME_MODEL), exist_ok= True) # tạo 1 folder
    dir_save_file = str(datetime.date.today())
    
    # lưu các biến khởi tạo vào 1 file
    if SAVE_CKPT :
        utils_save_cfg.save_cfg(cfg= cfg, checkpoint_dir= CHECKPOINT_DIR, name_model= NAME_MODEL, dir_save_file= dir_save_file)

    # khởi tạo data
    Dataset = Datasets(path_data= DATA_ROOT, load_width= LOAD_WIDTH,\
                          load_height= LOAD_HEIGHT, nb_classes= NUM_CLASSES)
    
    # chia dữ liệu làm 80% để huấn luyện    
    train_size = int(0.8 * len(Dataset))
    val_size = len(Dataset) - train_size
    trainDataset, valDataset = random_split(Dataset, [train_size, val_size]) # 0 1 2 3 4 5 6 7 8 9 , [8,2] 

    # chuyêmr dữ liệu sang 1 định dạng để máy đọc được
    # batch_size : nhóm các hình ảnh với nhau thành 1 tệp ( ví dụ 16 hình ảnh 1 lúc để học trong 1 lần)
    # 3 cái còn lại đọc docs or chatgpt
    trainLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, \
                             shuffle= True, num_workers= NUM_WORKERS) 
    valLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, \
                           shuffle= True, num_workers= NUM_WORKERS)
    dataset_sizes = {
        'train' : len(trainDataset),
        'val' : len(valDataset),
    }
    dataLoader = { 
        'train' : trainLoader,  
        'val': valLoader
    }

    print("device :", DEVICE) # cpu or gpu
    
    # 
    # if NAME_MODEL == 'resnet101':
    #     model = torchvision.models.resnet101(pretrained = True) # khởi tạo mô hình 
    #     num_features = model.fc.in_features
    #     model.fc = nn.Linear(num_features, 2)
    # else:
    #     model = utils_model.create_model(name_model= NAME_MODEL, num_classes= NUM_CLASSES)
    model = torchvision.models.vgg16()
    # num_features = model.classifier[6].in_features
    # model.classifier[6] = nn.Linear(num_features, 2)
    for param in model.features.parameters():
      param.requires_grad = False

  # Thay đổi lớp cuối cùng (classifier) để thích ứng với bài toán phân loại của bạn
    num_classes = 2  # Số lượng lớp đầu ra (ví dụ: xe máy và xe đạp)
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),  # Lớp Fully Connected thứ nhất
        nn.LeakyReLU(),
        nn.Dropout(0.5),  # Thêm Dropout với xác suất bị bỏ (dropout probability) là 0.5
        nn.Linear(4096, 2048),  # Lớp Fully Connected thứ hai
        nn.LeakyReLU(),  
        nn.Dropout(0.5),  # Thêm Dropout với xác suất bị bỏ là 0.5
        nn.Linear(2048, num_classes)  # Lớp đầu ra cho số lượng lớp của bạn
    )

    model.to(device= DEVICE) # chuyển mô hình sang sử dung gpu hay cpu
    print(model)
    model.train()
    criterion = utils_loss.create_loss(name_loss= NAME_LOSS, num_classes= NUM_CLASSES) # khởi tạo hàm tính hàm mất mát

    optimizer = torch.optim.Adam(model.parameters(), lr = LR,  weight_decay=WEIGHT_DECAY) # hàm tính đạo hàm
    
    lr_schedule_values = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # hàm để điều chỉnh learning rate

    best_acc = 0.0
    best_epoch = None
    logger.info('*'*30)
    logger.info('Start train: name model: {}, run times {}'.format(NAME_MODEL, dir_save_file))

    for epoch in range(1, EPOCHS +1): # bắt đầu huấn luyện với số lần dạy là : epochs

        # result = train_one_epoch(model= model, criterion= criterion, optimizer= optimizer, 
        #                          lr_schedule_values= lr_schedule_values,
        #                          device= DEVICE, epoch= epoch, epochs= EPOCHS, dataset_sizes= dataset_sizes,
        #                          dataLoader = dataLoader, num_save_ckpt= NUM_SAVE_CKPT,
        #                          save_ckpt= SAVE_CKPT, name_model= NAME_MODEL, train_on= TRAIN_ON, 
        #                          checkpoint_dir= CHECKPOINT_DIR, num_save_file= num_save_file, logger = logger)

        print(f'Epoch {epoch}/{EPOCHS}')
        print('-' * 20)
        for phase in ['train', 'val']: # chia làm 2 phần
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            for inputs in tqdm(dataLoader[phase]): # load từng tập dữ liệu 
                input = inputs['image'].to(DEVICE) # như model # 16 ảnh 16*3*224*224
                labels = inputs['label'].to(DEVICE)  # như model

                optimizer.zero_grad() # khởi tạo đạo hàm
                with torch.set_grad_enabled(phase == 'train'): # nếu đang huấn luyện bật tính toán đạo ghàm lên
                    outputs = model(input) # dự đoán
                    _, preds = torch.max(outputs, 1) # lấy kết quả dự đoán

                    loss = criterion(outputs, labels) # tính toán sai số, mất mát
                    if phase == 'train':
                        loss.backward() 
                        optimizer.step() # tímh đạo hàm và cập nhật lại cho mô hình
                running_loss += loss.item() * input.size(0) # tính tiongr sai số
                
                running_corrects += torch.sum(preds == labels.data) # tính tỉ lệ đúng
            if phase == 'train':
                lr_schedule_values.step() # cập nhật 1 số tham số khởi tạo , có thể bỏ đi
            epoch_loss = running_loss / dataset_sizes[phase] # tính loss trugn bình
            epoch_acc = running_corrects.double() / dataset_sizes[phase] # tính đúng trung bình
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            logger.info(f'Epoch {epoch} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and best_acc > epoch_acc: # nếu huấn luyện lần này có tỉ lệ học cao hơn thì lưu lại
                best_acc = epoch_acc # 10 90% , 11 95 % 12 90%
                best_model_wts = copy.deepcopy(model)

        # phần dưới lưu trọng số sau khi huấn luyện
        if (epoch % NUM_SAVE_CKPT ==0  or epoch == EPOCHS or epoch == 1) and SAVE_CKPT :
            if TRAIN_ON == 'ssh' :
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                }, os.path.join(CHECKPOINT_DIR, NAME_MODEL, \
                            dir_save_file, str(epoch)+'.pth'))
            else :
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                }, str(epoch) +'.pth')
        

    # model.load_state_dict(best_model_wts) # lưu epoch ma có tỉ lệ đúng cao nhất
    if SAVE_CKPT :
        if TRAIN_ON == 'ssh' :
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_model_wts.state_dict(),
            }, os.path.join(CHECKPOINT_DIR, NAME_MODEL, \
                            dir_save_file, ("best_epoch"+".pth"))) 
        else : 
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_model_wts.state_dict(),
            }, "best_epoch.pth")
        
def get_args_parser(): # 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type= str, default= 'train')
    parser.add_argument('--data_root', type= str, default='datasets/train')
    parser.add_argument('--checkpoint_dir', type= str, default= 'checkpoint')
    parser.add_argument('--name_model', type= str, default= 'vgg16')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    args = get_args_parser()
    start_time = time.time()
    train(args= args)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


