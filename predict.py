import argparse

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from datasets import load_dataset

def main(args):
    # Load datasets & dataloaders
    ds = load_dataset("sayakpaul/nyu_depth_v2")

    # Load model
    # checkpoint = "Intel/dpt-large"
    checkpoint = "vinvino02/glpn-nyu"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForDepthEstimation.from_pretrained(checkpoint)

    for x, y in tqdm(ds):
        pixel_values = image_processor(x, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = model(pixel_values)
            predicted_depth = outputs.predicted_depth
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=x.shape[2:],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        output = prediction.squeeze().cpu().numpy()[0]
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)
        depth.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str,
                        default="~/depth-estimation/NYUv2",
                        help="Directory where the nyu_depth_v2 labeled dataset will be extracted")
    main(parser.parse_args())
