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

train file arguments:
- **model**: Specifies which model to use for training. The default value is 'DCNN', but it can also be set to 'AUTOENC' or 'DCNN_2ENC'.
- **GAN**: A boolean flag that indicates whether to train the model using GAN loss or not. By default, this flag is set to False.
- **bottleneck**: Specifies which bottleneck architecture to use. The default value is 'DFA', but it can also be set to 'Vit' or 'None'.
- **loss**: Specifies which loss function to use during training. The default value is 'ROI', but it can also be set to 'ROI_SM' or 'L1'.
- **weighted**: A boolean flag that indicates whether to give more weight to samples that don't have PTVs. By default, this flag is set to False.
- **output_path**: Specifies the directory where the trained model and other output files will be saved.
- **PTV_estimate**: A boolean flag that indicates whether to train a model to estimate missing PTVs. By default, this flag is set to False.
- **batch_size**: Specifies the batch size for training. The default value is 32.
- **list_GPU_ids**: Specifies which GPUs to use for training. By default, only GPU 0 is used.
- **max_iter**: Specifies the maximum number of iterations.
- **epochs**: number of epochs, By default 50
- **iter-per-epoch**:
- **split-seed**: training validition split seed
- **without_distance**: distance transform of the ptvs with the inputs

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