import argparse
from pytorchnyuv2.nyuv2 import NYUv2
from torchvision import transforms


def main(args):
    t = transforms.Compose([transforms.RandomCrop(400), transforms.ToTensor()])
    NYUv2(root=args.root, download=True,
          rgb_transform=t, sn_transform=t, seg_transform=t, depth_transform=t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str,
                        default="~/depth-estimation/NYUv2",
                        help="Directory where the nyu_depth_v2 labeled dataset will be extracted")
    main(parser.parse_args())
