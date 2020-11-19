"""Model predict."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 11月 02日 星期一 17:53:20 CST
# ***
# ************************************************************************************/
#
import argparse
import glob
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from data import Video
from model import enable_amp, get_model, model_device, model_load

if __name__ == "__main__":
    """Predict."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default="models/VideoZoom.pth", help="checkpint file")
    parser.add_argument('--input', type=str,
                        default="dataset/predict/input", help="input folder")
    parser.add_argument('--output', type=str,
                        default="dataset/predict/output", help="output folder")
    args = parser.parse_args()

    model = get_model()
    model_load(model, args.checkpoint)
    device = model_device()
    model.to(device)
    model.eval()

    enable_amp(model)

    totensor = transforms.ToTensor()
    toimage = transforms.ToPILImage()

    video = Video()
    video.reset(args.input)
    progress_bar = tqdm(total=len(video))

    h = video.height
    w = video.width
    scale = 4

    count = 1
    for index in range(len(video)):
        progress_bar.update(1)

        # create input tensor, BxTxCxHxW
        input_tensor = video[index][1:3].unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor = model(input_tensor).clamp(0, 1.0).squeeze()
        # print(output_tensor.size()) TxCxHxW

        # Output result
        output_tensor = output_tensor.cpu()
        output_tensor = output_tensor[:, :, 0:scale*h, 0:scale*w]

        for k in range(2):
            toimage(output_tensor[k]).save(
                "{}/{:06d}.png".format(args.output, count))
            count += 1
