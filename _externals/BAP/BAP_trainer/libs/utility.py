import os
import random
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


from .classes import reg_dataset
from .densenet import densenet121
from .config import lr_init, sched_params

''' FUNCS '''

def get_DenseNet(device: torch.device = torch.device("cpu")):

    ''' 
        Returns DenseNet121 and sends it to device
    '''
    
    
    model = densenet121(mode = "classifier",
                        bn_size = 4, 
                        num_classes = 1,
                        in_channels = 1,
                        memory_efficient = False);
    model.to(device);
    return model



def get_device():
    
    if torch.cuda.is_available():
        print("\nIn use: GPU\n");
        return torch.device(torch.cuda.current_device()); 
    
    else:
        print("\nIn use: CPU\n");
        return torch.device("cpu");
    
    
    
def set_seed(seed: int = 82):
    np.random.seed(seed);
    torch.manual_seed(seed);
    torch.cuda.manual_seed(seed);
    torch.cuda.manual_seed_all(seed);
    torch.backends.cudnn.deterministic = True;
    random.seed(seed);
   
   
    
def get_preTrained_DenseNet(device: torch.device = torch.device("cpu"),
                             path: str = "./"):    
    mode = "classifier";
    nClasses = 1;
    model = densenet121(mode=mode,bn_size=4, num_classes=nClasses,
                        in_channels=1, memory_efficient=False);
    model.load_state_dict(torch.load(path, map_location = torch.device(device.type)));
    model.to(device);
    return model



def check_folder(path: str):
    
    ''' Check if folder exist or create it '''
    if not os.path.isdir(path):
        print(f"[!] {path} creata.")
        os.mkdir(path);
   
        
        
def save_curve(path: str,
               train_means: list,
               train_stds: list):
    train_data = np.array([train_means,train_stds]);
    np.save(f"{path}learning_curve.npy",train_data);
    
    

def save_model(path: str,
               model: nn.Module):
    torch.save(model.state_dict(), f"{path}/trained_model.pt");


    
def save_metadata(path: str,
                  train_losses: list,
                  test_losses: list,
                  train_ids: list,
                  test_ids: list,
                  train_targets: list,
                  test_targets: list,
                  train_preds: list,
                  test_preds: list,
                  ):

    df = pd.DataFrame();
    
    df["id"] = train_ids + test_ids;
    df["age"] = train_targets + test_targets;
    df["split"] = ["train"]*len(train_ids) + ["test"]*len(test_ids);
    df["pred"] = train_preds + test_preds;
    df["loss"] = train_losses + test_losses;
    
    df.to_csv(f"{path}performance.csv",
              index = False);
    
    

def load_dataframe(path: str):
    
    if path.endswith(".csv"):
        return pd.read_csv(path, dtype = str);
    
    else:
        return pd.read_csv(path, 
                           sep = "\t",
                           dtype = str);
    
    

def get_dataload(batch_size: int,
                data_path: str,
                dataset_path: str,
                _min = float,
                _max = float):
    
    data = load_dataframe(data_path);
    dataset = reg_dataset(dataset_path = dataset_path,
                          data = data,
                          _min = _min,
                          _max = _max);
    dload = DataLoader(dataset = dataset,
                       batch_size = batch_size,
                       shuffle = True);
    return dload;



def check_cuda(device = torch.device):
    return (device == torch.device("cpu"));



def get_training_params(model: nn.Module):
    
    criterion = nn.L1Loss(reduction = "none");
    optimizer = SGD(model.parameters(), lr = lr_init);
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer,
                                               T_0 = sched_params["T_0"],
                                               T_mult = sched_params["T_mult"],
                                               eta_min = sched_params["eta_min"]);
    return criterion, optimizer, lr_scheduler;


def get_label(age: float,
              th: float):
    
    if age < th:
        return f"age < {th}";
    else:
        return f"age >= {th}";