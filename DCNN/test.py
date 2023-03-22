# -*- encoding: utf-8 -*-
import os
import sys
import torch.nn as nn

if os.path.abspath("..") not in sys.path:
    sys.path.insert(0, os.path.abspath(".."))
import argparse

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/DCNN")
from DCNN.loss import L1Loss
import torch
from DataLoader.my_dataloader import get_loader
from model import Model, Model_sep, Ensemble
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="DCNN", help="AUTOENC, DCNN, DCNN_2ENC, ENSEMBLE"
    )
    parser.add_argument(
        "--GAN", default=False, action="store_true", help="Train with gan loss"
    )
    parser.add_argument("--bottleneck", default="DFA", help="DFA, Vit, NONE")
    parser.add_argument("--loss", default="ROI_SM", help="ROI, ROI_SM, L1")
    parser.add_argument(
        "--weighted",
        default=False,
        action="store_true",
        help="give more weights to the samples that don't have ptvs",
    )
    parser.add_argument("--output_path", default="Output", help="")
    parser.add_argument(
        "--without_distance",
        default=False,
        action="store_true",
        help="distance transform of the ptvs as input",
    )
    args = parser.parse_args()

    exten = ""
    gan = "_"
    bott = "_" + args.bottleneck

    if args.without_distance:
        exten = "_WD"
    if args.GAN:
        gan = "_GAN"

    if args.model != "ENSEMBLE":
        out_dir = (
            args.output_path + "/" + args.model + gan + bott + "_" + args.loss + exten
        )
    else:
        out_dir = args.output_path + "/" + args.model

    if not os.path.exists(out_dir):
        raise Exception(
            "Model does not exist, you can train a model under these settings"
        )
    else:
        predicted_test = out_dir + "/output"
        if not os.path.exists(predicted_test):
            os.mkdir(predicted_test)


    # Models input/output
    if args.without_distance:
        in_ch = 3
        out_ch = 1
    else:
        in_ch = 4
        out_ch = 1
    print(out_dir)
    model_PATH = out_dir + "/best_val_evaluation_uloss.pkl"

    # Models
    if args.model == "AUTOENC":
        model1 = Model(
            in_ch=in_ch,
            out_ch=out_ch,
            list_ch=[-1, 32, 64, 128, 256],
            Unet=False,
            bottleneck=args.bottleneck,
            with_dropout=False,
        )
        model1.load_state_dict(torch.load(model_PATH)["network_state_dict"])
        model2 = model1
    elif args.model == "DCNN":
        model1 = Model(
            in_ch=in_ch,
            out_ch=out_ch,
            list_ch=[-1, 32, 64, 128, 256],
            bottleneck=args.bottleneck,
            with_dropout=False,
        )
        model1.load_state_dict(torch.load(model_PATH)["network_state_dict"])
        model2 = model1
        
        args.model = args.model + "_" + args.bottleneck
    elif args.model == "DCNN_2ENC":
        model1 = Model_sep(
            in_ch=in_ch,
            out_ch=out_ch,
            list_ch=[-1, 32, 64, 128, 256],
            bottleneck=args.bottleneck,
        )
        model1.load_state_dict(torch.load(model_PATH)["network_state_dict"])
        model2 = model1
    elif args.model == "DCNN_2ENC_AUTO":
        model1 = Model_sep(
            in_ch=in_ch,
            out_ch=out_ch,
            list_ch=[-1, 32, 64, 128, 256],
            bottleneck=args.bottleneck,
            Unet=False,
        )
        model1.load_state_dict(torch.load(model_PATH)["network_state_dict"])
        model2 = model1
    elif args.model == "ENSEMBLE":
        paths = [
            "Output/DCNN__DFA_ROI_SM/best_val_evaluation_uloss.pkl",
            "Output/DCNN__DFA_ROI/best_val_evaluation_uloss.pkl",
            "Output/DCNN__Vit_ROI_SM/best_val_evaluation_uloss.pkl",
            "Output/DCNN_GAN_DFA_ROI_SM/best_val_evaluation_uloss.pkl",
            "Output/DCNN_2ENC__DFA_ROI_SM/best_val_evaluation_uloss.pkl",
            "Output/AUTOENC__DFA_ROI_SM/best_val_evaluation_uloss.pkl"
        ]
        models = []

        model1 = Model(in_ch=4, out_ch=1, list_ch=[-1, 32, 64, 128, 256], bottleneck="DFA")
        model2 = Model(in_ch=4, out_ch=1, list_ch=[-1, 32, 64, 128, 256], bottleneck="DFA")
        model3 = Model(in_ch=4, out_ch=1, list_ch=[-1, 32, 64, 128, 256], bottleneck="Vit")
        model4 = Model(in_ch=4, out_ch=1, list_ch=[-1, 32, 64, 128, 256], bottleneck="DFA")

        model5 = Model_sep(in_ch=4, out_ch=1, list_ch=[-1, 32, 64, 128, 256])

        model6 = Model(
            in_ch=4, out_ch=1, list_ch=[-1, 32, 64, 128, 256], Unet=False, bottleneck="DFA"
        )

        model1.load_state_dict(torch.load(paths[0])["network_state_dict"])
        models.append(model1)
        model2.load_state_dict(torch.load(paths[1])['network_state_dict'])
        models.append(model2)
        model3.load_state_dict(torch.load(paths[2])['network_state_dict'])
        models.append(model3)
        model4.load_state_dict(torch.load(paths[3])['network_state_dict'])
        models.append(model4)
        model5.load_state_dict(torch.load(paths[4])["network_state_dict"])
        models.append(model5)
        model6.load_state_dict(torch.load(paths[5])["network_state_dict"])
        models.append(model6)
        model1 = Ensemble(models)
        model2 = Ensemble([models[4],models[5]])
    else:
        raise Exception("Model does not exist")
    
    model1 = model1.eval()
    model2 = model2.eval()

    # Data loader
    test_loader = get_loader(
        "data",
        test_bs=1,
        test_num_samples_per_epoch=1200,
        num_works=1,
        with_miss_PTVs=True,
        with_distance=True,
        test=True,
    )


    for batch_idx, list_loader_output in tqdm(enumerate(test_loader)):
        path = list_loader_output[1][0]
        input_ = list_loader_output[0][0]
        target = list_loader_output[0][1][0, 0]
        mask = list_loader_output[0][2][0][0]
        if input_[0][0].sum() != 0:
            output = model1(input_)[0]
            img_pred = (output * mask).detach().numpy()
        else:
            output = model2(input_)[0]
            img_pred = (output * mask).detach().numpy()

        np.save(predicted_test + "/" + path.split("/")[-1], img_pred)

  