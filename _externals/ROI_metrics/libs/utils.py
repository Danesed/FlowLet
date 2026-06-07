import os
import torch
import numpy as np
import nibabel as nib
from scipy.special import kl_div


def check_folder(path: str):
    
    if not os.path.isdir(path):
        print(f"[!] {path} created.")
        os.mkdir(path);
        

def extract_age(string: str):
    words = string.split(sep = "_");  
    age_idx = words.index("AGE") + 1;
    age = float(words[age_idx]);
    return age;


def load_mgz(path: str):
    img = nib.load(path);
    data = img.get_fdata();
    return data;


def normalize(array: np.ndarray,
              _min : float,
              _max : float):
    norm = (array - _min)/(_max - _min);
    return np.clip(norm,
                   a_min = 0,
                   a_max = 1);


def get_normalized_hist(arr: np.ndarray):
    V = len(arr);
    h = np.histogram(arr, bins = range(256));
    return h[0]/V;
    
      
def MAE(arr1: np.ndarray,
        arr2: np.ndarray):
    V = len(arr1);
    return np.sum(np.abs(arr1 - arr2))/V;


def MSE(arr1: np.ndarray,
        arr2: np.ndarray):
    V = len(arr1);
    return np.sum((arr1 - arr2)**2)/V;


def KLD(P: np.ndarray,
        Q: np.ndarray):
    eps = 1e-5;
    lg = np.log2(P/(Q + eps) + eps);
    return sum(P*lg);


def get_torch_normalized_hist(arr: torch.tensor):
    V = torch.numel(arr);
    h = torch.histogram(arr.cpu().float(), bins = (torch.tensor(range(257)) - 0.5));
    return h[0]/V;


def torch_KLD(P: torch.tensor,
        Q: torch.tensor):
    eps = 1e-5;
    lg = torch.log2(P/(Q + eps) + eps)
    return torch.sum(P*lg);


def get_device():

    if torch.cuda.is_available():
        print("\nUsing: GPU\n");
        return torch.device(torch.cuda.current_device());    
    else:
        print("\nUsing: CPU\n");
        return torch.device("cpu");
    
def extract_flowlet_model(string: str):
    words = string.split(sep = "_");  
    return f"FlowLet: {words[3]}";