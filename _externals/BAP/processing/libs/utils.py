import os
import sys
import numpy as np
import nibabel as nib
from collections import Counter
from scipy.stats import pearsonr

def nii2npy(path: str):
    img = nib.load(path);
    return img.get_fdata();

def standardize(data: np.ndarray):
    return (data - data.mean())/data.std();

def extract_img_metadata(img: np.ndarray):
    d = {};
    d["min"] = img.min();
    d["max"] = img.max();
    d["mean"] = img.mean();
    d["std"] = img.std();
    return d;

def extract_filename_metadata(string: str):
    words = string.split(sep = "_");

    d = {};    
    if "ADNI" in words[0]:
        d["dataset"] ="ADNI";
    elif "OASIS" in words[0]: 
        d["dataset"] = "OASIS";
    elif "OPENBHB" in words[0]:
        d["dataset"] = "OPENBHB";
    else:
        sys.exit(f"[X] Dataset not found (string: {string})");
    
    age_idx = words.index("AGE") + 1;
    d["age"] = float(words[age_idx]);
    
    return d;


def extract_age(string: str):
    words = string.split(sep = "_");  
    age_idx = words.index("AGE") + 1;
    age = float(words[age_idx]);
    return age;

def correlation(a, b):
    return pearsonr(a.flatten(),b.flatten())[0];


def get_all_rotations(data):
    rotations = [(data, "original")];
    for axis in [(0, 1), (0, 2), (1, 2)]:
        for k in range(1, 4):
            rotated = np.rot90(data, k=k, axes=axis);
            rotations.append((rotated, f"axes={axis}, k={k}"));
    return rotations;


def find_best_rotation(generated_data, reference_data):
    best_score = -np.inf;
    best_rotation = generated_data;
    best_descr = "original";
    for rotated, descr in get_all_rotations(generated_data):
        score = correlation(rotated, reference_data);
        if score > best_score:
            best_score = score;
            best_rotation = rotated;
            best_descr = descr;
    return best_rotation, best_descr, best_score;


def get_edges(big_shape: tuple, small_shape: tuple):
    edges = [];
    for i in range(3):
        _mean = int(np.floor((big_shape[i] - small_shape[i])/2));
        edges.append(_mean);
        edges.append(big_shape[i] - (_mean + 1));
    return edges;

def pad(array: np.ndarray, final_shape: tuple):
    original_shape = array.shape;    
    edges = get_edges(small_shape = original_shape, 
                    big_shape = final_shape);

    padded = np.zeros(final_shape);
    padded[edges[0]:edges[1],edges[2]:edges[3],edges[4]:edges[5]] = array;
    
    return padded;


def crop(array: np.ndarray, final_shape: tuple):
    original_shape = array.shape;
    edges = get_edges(small_shape = final_shape, 
                    big_shape = original_shape);
    return array[edges[0]:edges[1],edges[2]:edges[3],edges[4]:edges[5]];


def load_training_sample(path: str):
    sample = np.load(path);
    return np.squeeze(sample, axis = (0,1));


def guess_age(real_ages: list, num_samples: int):
    counts = Counter(real_ages);

    age_nums = list(counts.keys());
    freqs = list(counts.values());

    prob = [f / sum(freqs) for f in freqs];

    return np.random.choice(age_nums,
                            size = num_samples,
                            p = prob);
    
def check_folder(path: str):
    
    if not os.path.isdir(path):
        print(f"[!] {path} creata.")
        os.mkdir(path);