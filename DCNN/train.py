# -*- encoding: utf-8 -*-
import os
import sys
import torch.nn as nn
if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))
import argparse
sys.path.append('/home/infres/ahmad-21/Master_MVA/DL_MI_project/')
sys.path.append('/home/infres/ahmad-21/Master_MVA/DL_MI_project/DCNN')
import torch
import copy

from DataLoader.my_dataloader import get_loader
from NetworkTrainer.network_trainer import NetworkTrainer
from model import Model, EMA, Discriminator, Model_sep
from loss import Loss, L1Loss, Loss_ROI, Loss_SM_ROI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='DCNN',
                        help='AUTOENC, DCNN, DCNN_2ENC')
    parser.add_argument('--GAN', default=False, action='store_true',
                        help='Train with gan loss')
    parser.add_argument('--bottleneck', default='DFA',
                        help='DFA, Vit, None')                    
    parser.add_argument('--loss', default='ROI_LOSS',
                        help='ROI_LOSS, ROI_SM_LOSS, L1LOSS, WEIGHTED_ROI_LOSS')
    parser.add_argument('--weighted', default=False, action='store_true',
                        help="give more weights to the samples that don't have ptvs")
    parser.add_argument('--output_path', default='Output',
                        help='')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for training (default: 32)')
    parser.add_argument('--list_GPU_ids', nargs='+', type=int, default=[0],
                        help='list_GPU_ids for training (default: [0])')
    parser.add_argument('--max_iter', type=int, default=100000,
                        help='training iterations(default: 100000)')
    parser.add_argument('--iter-per-epoch', type=int, default=1000,
                        help='')
    parser.add_argument('--split-seed', type=int, default=10,
                        help='training validition split seed')
    parser.add_argument('--with_distance', default=True, action='store_true',
                        help='add distance transform of the ptvs as input')
    parser.add_argument('--wandb', default=False, action='store_true',
                        help='training iterations(default: False)')
    args = parser.parse_args()

    #  Start training
    trainer = NetworkTrainer()
    trainer.setting.project_name = args.model
    trainer.setting.output_dir = args.output_path + '/' + args.model
    if not os.path.exists(trainer.setting.output_dir):
        os.mkdir(trainer.setting.output_dir)

    trainer.setting.wandb = args.wandb
    trainer.setting.EMA = None
    list_GPU_ids = args.list_GPU_ids

    if args.model=='AUTOENC':
        trainer.setting.network = Model(in_ch=4, out_ch=1,
                                        list_ch=[-1, 32, 64, 128, 256], Unet=False, bottleneck=args.bottleneck)
    elif args.model=='DCNN':
        trainer.setting.network = Model(in_ch=4, out_ch=1,
                                        list_ch=[-1, 32, 64, 128, 256], bottleneck=args.bottleneck)
    elif args.model=='DCNN_2ENC':
        trainer.setting.network = Model_sep(in_ch=4, out_ch=1,
                                        list_ch=[-1, 32, 64, 128, 256])
    if args.GAN:
        trainer.setting.is_GAN = True
        trainer.setting.Discriminator = Discriminator(1)
        trainer.setting.criterion = nn.BCELoss()

    if args.loss=='ROI_LOSS':
        trainer.setting.loss_function = Loss_ROI()
    elif args.loss=='ROI_SM_LOSS':
        trainer.setting.loss_function = Loss_SM_ROI()
    elif args.loss=='L1LOSS':
        trainer.setting.loss_function = L1Loss()

    trainer.setting.ema_network = copy.deepcopy(trainer.setting.network)
    

    trainer.set_GPU_device(list_GPU_ids)

    trainer.setting.network.to(trainer.setting.device)
    trainer.setting.ema_network.to(trainer.setting.device)


    trainer.setting.max_iter = args.max_iter

    trainer.setting.train_loader, trainer.setting.val_loader, _ = get_loader(
        train_bs=args.batch_size,
        val_bs=1,
        train_num_samples_per_epoch=args.batch_size*args.iter_per_epoch,  
        val_num_samples_per_epoch=1200,
        num_works=1,
        resplit=True,
        seed=args.split_seed,
        with_miss_PTVs=True,
        with_distance=args.with_distance
        )

    trainer.setting.eps_train_loss = 0.01
    trainer.setting.lr_scheduler_update_on_iter = True
    trainer.setting.weighted_loss = args.weighted
    trainer.setting.loss_function1 = L1Loss()
    trainer.setting.criterion = nn.BCELoss()
    
    trainer.set_optimizer(optimizer_type='Adam',
                          args={
                              'lr_encoder': 5e-4,
                              'lr_decoder': 5e-4,
                              'lr': 5e-4,
                              'weight_decay': 1e-4
                          }
                          )

    trainer.set_lr_scheduler(lr_scheduler_type='cosine',
                             args={
                                 'T_max': args.max_iter,
                                 'eta_min': 1e-7,
                                 'last_epoch': -1
                             }
                             )

    if not os.path.exists(trainer.setting.output_dir):
        os.mkdir(trainer.setting.output_dir)
    
    trainer.run()

    trainer.print_log_to_file('# Done !\n', 'a')
