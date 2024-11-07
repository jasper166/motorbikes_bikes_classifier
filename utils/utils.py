import os
import torch
import argparse

def write_txt(noidung, path, remove = False, space = True):
    if os.path.exists(path) and remove == True:
        os.remove(path)
    if os.path.exists(os.path.join(*path.split('/')[:-1])) == False :
        os.makedirs(os.path.join(*path.split('/')[:-1]), exist_ok= True)
    with open(path, 'a') as f:
        if space :
            f.write("%s\n" % noidung)
        else: 
            f.write("%s "% noidung)
        f.close()

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise ValueError("'{}' is not a valid checkpoint path".format(checkpoint_path))
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

