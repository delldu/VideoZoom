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

from model import get_model, model_load, model_setenv
from data import Video

if __name__ == "__main__":
    """Predict."""

    model_setenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default="models/VideoZoom.pth", help="checkpint file")
    parser.add_argument('--input', type=str, default="dataset/predict/input", help="input folder")
    parser.add_argument('--output', type=str, default="dataset/predict/output", help="output folder")
    args = parser.parse_args()

    # CPU or GPU ?
    device = torch.device(os.environ["DEVICE"])

    model = get_model()
    model_load(model, args.checkpoint)
    model.to(device)
    model.eval()

    if os.environ["ENABLE_APEX"] == "YES":
        from apex import amp
        model = amp.initialize(model, opt_level="O1")

    totensor = transforms.ToTensor()
    toimage = transforms.ToPILImage()

    video = Video()
    video.reset(args.input)
    progress_bar = tqdm(total=len(video))

    count = 1
    for index in range(len(video)):
        progress_bar.update(1)

        # create input tensor, BxTxCxHxW
        input_tensor = video[index][0:2].unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor = model(input_tensor).clamp(0, 1.0).squeeze()
        # print(output_tensor.size())

        # Output result
        for k in range(2):
            toimage(output_tensor[k].cpu()).save(
                "{}/{:06d}.png".format(args.output, count))
            count += 1

         # image = Image.open(filename).convert("RGB")
         #  input_tensor = totensor(image).unsqueeze(0).to(device)

         #   with torch.no_grad():
         #        output_tensor = model(input_tensor).clamp(0, 1.0).squeeze()

         #    # xxxx--modify here
         #    toimage(output_tensor.cpu()).save(
         #        "{}/{:06d}.png".format(args.output, index + 1))
