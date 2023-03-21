import torch.utils.data as data
import os
import random
import numpy as np
import sys
import cv2
sys.path.append('DataLoader')
from augmentation import \
    random_flip_2d,  random_rotate_around_z_axis, random_translate, to_tensor

import skimage.morphology
from scipy.ndimage import distance_transform_edt

def read_data(patient_dir):
    dict_images = {}
    list_structures = ['ct',
                       'possible_dose_mask',
                       'structure_masks',
                       'dose']

    for structure_name in list_structures:
        structure_file = patient_dir + '/' + structure_name + '.npy'

        if os.path.exists(structure_file):
            data = np.float32(np.load(structure_file))
            dict_images[structure_name] = data
        else:
            dict_images[structure_name] = np.zeros((1, 128, 128), np.uint8)

    return dict_images


def pre_processing(dict_images, with_distance=True, PTV_estimate=False):

    # CT image
    CT = dict_images['ct']
    CT = np.clip(CT, a_min=-1024, a_max=1500)
    CT = CT.astype(np.float32) / 1000.
    

    # Dose
    dose = dict_images['dose']# / 70.

    # Possible_dose_mask, the region that can receive dose
    possible_dose_mask = dict_images['possible_dose_mask']

    structure_masks = dict_images['structure_masks'] 

    PTVs = structure_masks[-3:]
    OARs = structure_masks[:-3]

    OAR_all = np.zeros((1, 128, 128), np.uint8)
    for OAR_i in range(7):
        OAR = OARs[OAR_i][None]
        OAR_all[OAR > 0] = OAR_i + 1
        
    if not PTV_estimate:
        PTVs = 70.0 / 70. * PTVs[2] \
            + 63.0 / 70. * PTVs[1] \
            + 56.0 / 70. * PTVs[0]
        if with_distance:
            
            distance_image = get_distance_image(PTVs)[None]
            
            list_images = [np.concatenate((PTVs[None], OAR_all, CT[None], distance_image), axis=0),  # Input
                        dose[None],  # Label
                        possible_dose_mask[None],
                        structure_masks]
        else:
            list_images = [np.concatenate((PTVs[None], OAR_all, CT[None]), axis=0),  # Input
                        dose[None],  # Label
                        possible_dose_mask[None],
                        structure_masks]
    
    else:
        PTVs = np.concatenate((PTVs, (1*(PTVs.sum(axis=0)==0)*possible_dose_mask)[None], (1*(possible_dose_mask==0))[None]), axis=0)
        
        list_images = [np.concatenate((CT[None], OAR_all), axis=0),  # Input
                    PTVs,  # Label
                    possible_dose_mask[None],
                    dose[None]]

    #list_images = [np.concatenate((OAR_all, CT[None]), axis=0),  # Input
    #               dose[None],  # Label
    #               possible_dose_mask[None]]

    return list_images

def train_transform(list_images):
    #list_images = [Input, Label(gt_dose), possible_dose_mask]
    #Random flip
    list_images = random_flip_2d(list_images, list_axis=[1], p=1.0)

    # Random rotation
    list_images = random_rotate_around_z_axis(list_images,
                                              list_angle=(0, 40, 80, 120, 160, 200, 240, 280, 320),
                                              list_boder_value=(0, 0, 0, 0),
                                              list_interp=(cv2.INTER_NEAREST, cv2.INTER_NEAREST, cv2.INTER_NEAREST, cv2.INTER_NEAREST),
                                              p=0.3)

    
    # Random translation, but make use the region can receive dose is remained
    list_images = random_translate(list_images,
                                   roi_mask=list_images[2][0, :, :],  # the possible dose mask
                                   p=0.8,
                                   max_shift=20,
                                   list_pad_value=[0, 0, 0, 0])

    list_images = to_tensor(list_images)

    return list_images


def val_transform(list_images):
    list_images = to_tensor(list_images)
    return list_images


def get_distance_image(mask):
    mask_erode = skimage.morphology.binary_erosion(mask)
    surface = np.uint8(mask) - np.uint8(mask_erode)
    distance = distance_transform_edt(np.logical_not(surface))
    distance[mask > 0] = -1 * distance[mask > 0]

    return distance.astype(np.float)


class MyDataset(data.Dataset):

    def __init__(self, dir, num_samples_per_epoch,
                phase, resplit=False,
                seed=10, 
                train_size=0.8, 
                with_miss_PTVs=True,
                with_distance=True,
                PTV_estimate=False):
        self.phase = phase
        self.with_miss_PTVs = with_miss_PTVs
        self.num_samples_per_epoch = num_samples_per_epoch
        self.with_distance = with_distance
        self.PTV_estimate = PTV_estimate
        self.transform = {'train': train_transform, 'val': val_transform, 'test': val_transform}
        if resplit:
            def condition(b):
                return 'sample' in b and ('train' in b or 'val' in b)
            train_val = [a[0] for a in os.walk(dir) if condition(a[0])]
            random.seed(seed)
            random.shuffle(train_val)
            length = len(train_val)
            split = int(length*train_size)

            self.list_case_id = {'train': train_val[:split],
                                'val': train_val[split:],
                                'test': [a[0] for a in os.walk(dir+'/test/') if 'sample' in a[0]]}[phase]
                         
        else:
            self.list_case_id = {'train': [a[0] for a in os.walk(dir+'/train/') if 'sample' in a[0]],
                                'val': [a[0] for a in os.walk(dir+'/validation/') if 'sample' in a[0]],
                                'test': [a[0] for a in os.walk(dir+'/test/') if 'sample' in a[0]]}[phase]
            
        if phase=='train':
            random.shuffle(self.list_case_id)
        self.sum_case = len(self.list_case_id)

    def __getitem__(self, index_):
        
        if index_ <= self.sum_case - 1:
            case_id = self.list_case_id[index_]
        else:
            new_index_ = index_ - (index_ // self.sum_case) * self.sum_case
            case_id = self.list_case_id[new_index_]

        # Randomly pick a slice as input

        dict_images = read_data(case_id)
        list_images = pre_processing(dict_images, with_distance=self.with_distance, PTV_estimate=self.PTV_estimate)
        
        if self.phase=='train' and  not self.PTV_estimate and \
        (list_images[2][0, :, :].sum()==0 or list_images[0][0, :, :].sum()==0):
            if list_images[2][0, :, :].sum()==0:
                return self.__getitem__(np.random.randint(self.__len__()))
            if list_images[0][0, :, :].sum()==0 and not self.with_miss_PTVs:
                return self.__getitem__(np.random.randint(self.__len__()))
                
        elif self.PTV_estimate and self.phase=='train':
            if list_images[1][0:3, :, :].sum()==0 or list_images[2][0, :, :].sum()==0:
                return self.__getitem__(np.random.randint(self.__len__()))


        list_images = self.transform[self.phase](list_images)
        
        if self.phase=='test':
            return list_images, case_id
        else:
            if list_images[0][0, :, :].sum()==0 or list_images[0][1, :, :].sum()==0:
                return {'data' : list_images, 'clas': 0}, case_id
            else:
                return {'data' : list_images, 'clas': 1}, case_id
    

    def __len__(self):
        return self.num_samples_per_epoch




def get_loader(dir,
                train_bs=1,
                val_bs=1,
                test_bs=1,
                train_num_samples_per_epoch=1, 
                val_num_samples_per_epoch=1, 
                test_num_samples_per_epoch=1, 
                num_works=0, 
                resplit=True,
                seed=10,
                train_size=0.9,
                with_miss_PTVs=True,
                with_distance=True,
                PTV_estimate=False,
                test=False):

    test_dataset = MyDataset(dir, num_samples_per_epoch=test_num_samples_per_epoch,
                            phase='test', 
                            resplit=resplit,  
                            seed=seed,
                            train_size=train_size,
                            with_miss_PTVs=with_miss_PTVs,
                            with_distance=with_distance,
                            PTV_estimate=PTV_estimate)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=test_bs, shuffle=False, num_workers=num_works,
                                 pin_memory=False)
    if test:
        return test_loader

        
    train_dataset = MyDataset(dir, num_samples_per_epoch=train_num_samples_per_epoch,
                            phase='train', 
                            resplit=resplit, 
                            seed=seed,
                            train_size=train_size,
                            with_miss_PTVs=with_miss_PTVs,
                            with_distance=with_distance,
                            PTV_estimate=PTV_estimate)


    val_dataset = MyDataset(dir, num_samples_per_epoch=val_num_samples_per_epoch,
                            phase='val', 
                            resplit=resplit,  
                            seed=seed,
                            train_size=train_size,
                            with_miss_PTVs=with_miss_PTVs,
                            with_distance=with_distance,
                            PTV_estimate=PTV_estimate)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=True, num_workers=num_works,
                                   pin_memory=False)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=val_bs, shuffle=False, num_workers=num_works,
                                 pin_memory=False)
    

    return train_loader, val_loader, test_loader