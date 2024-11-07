import torch
config = {
    'train': dict(
    BATCH_SIZE = 4,
    EPOCHS = 20,
    TRAIN_ON = 'ssh',
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),

    #model

    NUM_CLASSES = 2,
    LR = 1e-5, # 0.0001
    NUM_WORKERS = 2,
    NAME_LOSS = 'Poly1CrossEntropyLoss',
    WEIGHT_DECAY = 5e-4, 
    MOMENTUM = 0.9,
    PIN_MEMORY = True,
    
    #ckpt
    NUM_SAVE_CKPT = 1,
    SAVE_CKPT = True,

    #data
    RESIZE = True,
    LOAD_WIDTH = 224,
    LOAD_HEIGHT = 224,

    ),
    'test' : dict(
    NUM_CLASSES = 2,
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    RESIZE = True,
    LOAD_WIDTH = 224,
    LOAD_HEIGHT = 224,
    )
}