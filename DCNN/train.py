# -*- encoding: utf-8 -*-
import os
import sys
import torch.nn as nn

if os.path.abspath("..") not in sys.path:
    sys.path.insert(0, os.path.abspath(".."))
import argparse

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/DCNN")
import torch
import wandb
from DataLoader.my_dataloader import get_loader
from NetworkTrainer.network_trainer import NetworkTrainer
from model import Model, Discriminator, Model_sep, Ensemble
from loss import L1Loss, Loss_ROI, Loss_SM_ROI, PTV_estimator_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="DCNN", help="AUTOENC, DCNN, DCNN_2ENC")
    parser.add_argument(
        "--GAN", default=False, action="store_true", help="Train with gan loss"
    )
    parser.add_argument("--bottleneck", default="DFA", help="DFA, Vit, None")
    parser.add_argument("--loss", default="ROI_SM", help="ROI, ROI_SM, L1")
    parser.add_argument(
        "--weighted",
        default=False,
        action="store_true",
        help="give more weights to the samples that don't have ptvs",
    )
    parser.add_argument("--output_path", default="Output", help="")
    parser.add_argument(
        "--PTV_estimate",
        default=False,
        action="store_true",
        help="train a model to estimate the missing PTVs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size for training (default: 32)",
    )
    parser.add_argument(
        "--list_GPU_ids",
        nargs="+",
        type=int,
        default=[1],
        help="list_GPU_ids for training (default: [1])",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100000,
        help="training iterations(default: 100000)",
    )
    parser.add_argument("--epochs", type=int, default=40, help="epochs(default: 50)")
    parser.add_argument("--iter-per-epoch", type=int, default=1000, help="")
    parser.add_argument(
        "--split-seed", type=int, default=None, help="training validition split seed"
    )
    parser.add_argument(
        "--without_distance",
        default=False,
        action="store_true",
        help="distance transform of the ptvs as input",
    )
    parser.add_argument(
        "--wandb", default=False, action="store_false", help="weights and biases log"
    )
    parser.add_argument(
        "--train_size",
        default=0.9,
        help="read it only is split-seed is given otherwise the given split will be used",
    )
    parser.add_argument(
        "--without_PTV",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()

    #  Start training
    trainer = NetworkTrainer()
    trainer.setting.project_name = args.model

    exten = ""
    gan = "_"
    bott = "_" + args.bottleneck
    if args.PTV_estimate:
        exten = "_ptv1"
        
    if args.without_PTV:
        exten = "_wPTV"
    if args.without_distance:
        exten = "_WD"
    if args.GAN:
        gan = "_GAN"

    trainer.setting.output_dir = (
        args.output_path + "/" + args.model + gan + bott + "_" + args.loss + exten
    )

    if not os.path.exists(trainer.setting.output_dir):
        os.mkdir(trainer.setting.output_dir)

    if args.split_seed is None:
        resplit = False
        args.split_seed = -1
    else:
        resplit = True

    # Add result to weights and biases
    if args.wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="Radiation Dose Prediction",
            # track hyperparameters and run metadata
            config={
                "architecture": args.model,
                "bottleneck": args.bottleneck,
                "loss": args.loss,
                "seed": args.split_seed,
                "dataset": "OpenKBP",
                "epochs": 20,
                "with_distance": not args.without_distance,
                "GAN": args.GAN,
                "with_dropout": True,
            },
        )

    trainer.setting.wandb = args.wandb
    list_GPU_ids = args.list_GPU_ids
    trainer.set_GPU_device(list_GPU_ids)
    trainer.setting.max_epoch = args.epochs

    # Models input/output
    if args.without_distance:
        in_ch = 3
        out_ch = 1
    elif args.PTV_estimate:
        in_ch = 2
        out_ch = 5
    elif args.without_PTV:
        in_ch = 2
        out_ch = 1
    else:
        in_ch = 4
        out_ch = 1

    # Models
    if args.model == "AUTOENC":
        trainer.setting.network = Model(
            in_ch=in_ch,
            out_ch=out_ch,
            list_ch=[-1, 32, 64, 128, 256],
            Unet=False,
            bottleneck=args.bottleneck,
            with_dropout=False,
            PTV_estimator=args.PTV_estimate,
        )
    elif args.model == "DCNN":
        trainer.setting.network = Model(
            in_ch=in_ch,
            out_ch=out_ch,
            list_ch=[-1, 32, 64, 128, 256],
            bottleneck=args.bottleneck,
            with_dropout=False,
            PTV_estimator=args.PTV_estimate,
        )
        args.model = args.model + "_" + args.bottleneck
    elif args.model == "DCNN-2ENC":
        trainer.setting.network = Model_sep(
            in_ch=in_ch,
            out_ch=out_ch,
            list_ch=[-1, 32, 64, 128, 256],
            bottleneck=args.bottleneck,
        )
    elif args.model == "DCNN-2ENC-AUTO":
        trainer.setting.network = Model_sep(
            in_ch=in_ch,
            out_ch=out_ch,
            list_ch=[-1, 32, 64, 128, 256],
            bottleneck=args.bottleneck,
            Unet=False,
        )

    trainer.setting.network.to(trainer.setting.device)

    # Loss function
    if args.GAN:
        trainer.setting.is_GAN = True
        trainer.setting.Discriminator = Discriminator(1).to(trainer.setting.device)
        trainer.setting.criterion = nn.BCELoss()
    if not args.PTV_estimate:
        if args.loss == "ROI":
            trainer.setting.loss_function = Loss_ROI()
        elif args.loss == "ROI_SM":
            trainer.setting.loss_function = Loss_SM_ROI()
        elif args.loss == "L1":
            trainer.setting.loss_function = L1Loss()
    else:
        trainer.setting.loss_function = PTV_estimator_loss(
            weights=torch.tensor([1, 1, 1, 1, 0]).to(trainer.setting.device)
        )

    # Data loaders
    trainer.setting.train_loader, trainer.setting.val_loader, _ = get_loader(
        "data",
        train_bs=args.batch_size,
        val_bs=1,
        train_num_samples_per_epoch=args.batch_size * args.iter_per_epoch,
        val_num_samples_per_epoch=1200,
        num_works=1,
        resplit=resplit,
        seed=args.split_seed,
        train_size=args.train_size,
        with_miss_PTVs=True,
        with_distance=not args.without_distance,
        PTV_estimate=args.PTV_estimate,
        without_PTVs=args.without_PTV
    )

    trainer.setting.max_iter = args.max_iter
    trainer.setting.eps_train_loss = 0.01
    trainer.setting.lr_scheduler_update_on_iter = True
    trainer.setting.weighted_loss = args.weighted
    trainer.setting.loss_function1 = L1Loss()

    trainer.set_optimizer(
        optimizer_type="Adam",
        args={"lr_encoder": 5e-4, "lr_decoder": 5e-4, "lr": 5e-4, "weight_decay": 1e-4},
    )

    trainer.set_lr_scheduler(
        lr_scheduler_type="cosine",
        args={"T_max": args.max_iter, "eta_min": 1e-7, "last_epoch": -1},
    )

    trainer.run()
    trainer.print_log_to_file("# Done !\n", "a")
