# -*- encoding: utf-8 -*-
import os
import sys
import torch.nn as nn
if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))

import numpy as np
from tqdm import tqdm
import sklearn.metrics
import shutil


sys.path.append('/home/infres/ahmad-21/Master_MVA/DL_MI_project/')
sys.path.append('/home/infres/ahmad-21/Master_MVA/DL_MI_project/uvcgan')

def get_data_paths(dir):
    data = [path[0] for path in os.walk('data') if 'sample' in path[0]]
    return data

def read_data(patient_dir, struct=None):
    dict_images = {}
    if struct is None:
        list_structures = ['ct',
                        'possible_dose_mask',
                        'structure_masks',
                        'dose']
    else:
        structure_file = patient_dir + '/' + struct + '.npy'
        if os.path.exists(structure_file):
            data = np.float32(np.load(structure_file))
            if struct=='ct':
                data = np.clip(data, a_min=-1024, a_max=1500)
                data = data.astype(np.float32) / 1000.
            return data
        else:
            raise('does not exist')
    

    for structure_name in list_structures:
        structure_file = patient_dir + '/' + structure_name + '.npy'
  

        if os.path.exists(structure_file):
            data = np.float32(np.load(structure_file))
            dict_images[structure_name] = data
        else:
            dict_images[structure_name] = np.zeros((128, 128), np.uint8)

    structure_masks = dict_images['structure_masks'] 

    PTVs = structure_masks[-3:]
    OARs = structure_masks[:-3]
    CT = dict_images['ct']

    OAR_all = np.zeros((128, 128), np.uint8)
    for OAR_i in range(7):
        OAR = OARs[OAR_i]
        OAR_all[OAR > 0] = OAR_i + 1
        
    PTVs = 70.0 / 70. * PTVs[2] \
        + 63.0 / 70. * PTVs[1] \
        + 56.0 / 70. * PTVs[0]

    
    list_images = np.concatenate((PTVs[None], OAR_all[None], CT[None], dict_images['possible_dose_mask'][None],dict_images['dose'][None]), axis=0)


    return list_images

def missing_PTVs(data_paths, which_set=None, indices=True):
    missing_masks = []
    
    for i, patient_dir in tqdm(enumerate(data_paths)):
        if which_set is not None:
            if which_set not in patient_dir :
                continue
        patient_data = read_data(patient_dir, 'structure_masks')
        PTVs = patient_data[-3:]
        mask = read_data(patient_dir, 'possible_dose_mask')
        if PTVs.sum()==0 and mask.sum()!=0:
            if indices:
                missing_masks.append(i)
            else:
                missing_masks.append(patient_dir)
    return missing_masks

def read_CT_scans(data_paths):
    CTs = np.zeros((len(data_paths), 128, 128))
    for i, patient_dir in tqdm(enumerate(data_paths)):
        CTs[i] = read_data(patient_dir, 'ct')
    return CTs
def read_masks(data_paths):
    CTs = np.zeros((len(data_paths), 128, 128))
    for i, patient_dir in tqdm(enumerate(data_paths)):
        CTs[i] = read_data(patient_dir, 'possible_dose_mask')
    return CTs

def read_all_data(data_paths):
    dic = np.zeros((len(data_paths), 5, 128, 128))
    print("reading all the data")
    for i, patient_dir in tqdm(enumerate(data_paths)):
        dic[i] = read_data(patient_dir)
    return dic

def delete_duplication(data_paths):
    all_samples = read_all_data(data_paths)

    print("Find unique samples")
    _, indexes = np.unique(all_samples, axis=0, return_index=True)
    duplicated_samples = [i for i in np.arange(len(data_paths)) if i not in indexes]
    print("Deleting duplicated sampeles")
    print(f"{len(duplicated_samples)} duplicated samples found")
    for sample_indice in tqdm(duplicated_samples):
        path = data_paths[sample_indice]
        sample_name = path.split('/')[-1]
        set_ = path.split('/')[-2]

        path_delete = './data_cleaned/' + set_ + '/' + sample_name
        shutil.rmtree(path_delete)


if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--batch_size', type=int, default=32,
    #                    help='batch size for training (default: 32)')
    #args = parser.parse_args()
    print('prepare data') 



