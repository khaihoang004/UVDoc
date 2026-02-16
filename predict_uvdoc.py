import argparse
import os
from os.path import join
import torch
import cv2
import numpy as np
from tqdm import tqdm

import model
import data_UVDoc
import utils


def save_img(tensor, path):
    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ======================
    # Load model
    # ======================
    net = model.UVDocnet(num_filter=32, kernel_size=5)
    ckpt = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(ckpt["model_state"])
    net.to(device)
    net.eval()

    # ======================
    # Dataset (benchmark)
    # ======================
    Dataset = data_UVDoc.UVDocDataset
    dataset = Dataset(
        data_path=args.uvdoc_benchmark,
        appearance_augmentation=[],
        geometric_augmentations=[],
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ======================
    # Output dirs
    # ======================
    out_tex = join(args.outdir, "uwp_texture")
    out_bm = join(args.outdir, "bm")
    os.makedirs(out_tex, exist_ok=True)
    os.makedirs(out_bm, exist_ok=True)

    # ======================
    # Predict
    # ======================
    for idx, (img, _, _, _) in enumerate(tqdm(loader, desc="Predicting")):
        img = img.to(device)

        grid2D, _ = net(img)
        uwp = utils.bilinear_unwarping(img, grid2D, utils.IMG_SIZE)

        fname = f"{idx:05d}.png"
        save_img(uwp[0], join(out_tex, fname))
        save_img(uwp[0], join(out_bm, fname))

    print("✅ Prediction finished")
    print("Output saved to:", args.outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uvdoc_benchmark", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./preds")
    args = parser.parse_args()

    main(args)
