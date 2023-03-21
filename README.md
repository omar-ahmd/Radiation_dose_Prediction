#   DiffsuionDet for object tracking
This code is based on the implementation of [RTDosePrediction](https://https://github.com/LSL000UD/), 
The main contribution of this work is to use various variations of the DCNN proposes in the aformentioned repository, such as bottelneck, 2encoders, new loss function, and finally an ensemble of various DCNN-based models.

# Data Preparation
Download 2D [openkbp](https://codalab.lisn.upsaclay.fr/my/datasets/download/d10c84c1-7824-4a9f-8693-fc3f79c759ce). And put them in the following structure:

```
<dataets_dir>
      │
      ├── train
            └── sample_<>
                  ├── ct.npy
                  ├── dose.npy
                  ├── possible_dose_masks.npy
                  └── structure_masks.npy
      ├── validation
      └── test    
      
```   


# Train
Single GPU training

```
cd <prb_dir>
$ python3 DCNN/train.py --batch_size 64 --list_GPU_ids 1 --wandb --model DCNN --bottleneck Vit  --epochs 40 --iter-per-epoch 1000 --loss ROI_SM
```

# Test
```
cd <prb_dir>
$ python3 DCNN/test.py --model ENSEMBLE --loss ROI_SM 
```


# Acknowledgement
A large part of the codes are borrowed from [RTDosePrediction](https://https://github.com/LSL000UD/RTDosePrediction), [uvcgan](https://github.com/LS4GAN/uvcgan), thanks for their excellent work!