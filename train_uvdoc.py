import argparse
import gc
import os

import torch
from tqdm import tqdm

import data_UVDoc
import model
import utils

def setup_data(args):
    """
    UVDoc only (no split)
    """
    UVDoc = data_UVDoc.UVDocDataset

    train_data = UVDoc(
        data_path=args.data_path_UVDoc,
        appearance_augmentation=args.appearance_augmentation,
        geometric_augmentations=args.geometric_augmentationsUVDoc,
    )

    trainloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Validation: same dataset, NO augmentation
    val_data = UVDoc(
        data_path=args.data_path_UVDoc,
        appearance_augmentation=[],
        geometric_augmentations=[],
    )

    valloader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return trainloader, valloader


def get_scheduler(optimizer, args, epoch_start):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(
            0, epoch + epoch_start - args.n_epochs
        ) / float(args.n_epochs_decay + 1)
        return lr_l

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda_rule
    )


def update_learning_rate(scheduler, optimizer):
    old_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    new_lr = optimizer.param_groups[0]["lr"]
    print(f"Learning rate: {old_lr:.7f} -> {new_lr:.7f}")
    return new_lr


def write_log_file(log_file_name, loss, epoch, lr, phase):
    with open(log_file_name, "a") as f:
        f.write(
            f"\n{phase} | Epoch {epoch:03d} | "
            f"LR {lr:.6f} | MSE {loss:.6f}"
        )


def main_worker(args):
    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Using device:", device)

    # ---- Data ----
    trainloader, valloader = setup_data(args)

    # ---- Model ----
    UVDocnet = model.UVDocnet(num_filter=32, kernel_size=5).to(device)

    # ---- Loss ----
    criterion_L1 = torch.nn.L1Loss()
    criterion_MSE = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        UVDocnet.parameters(), lr=args.lr, betas=(0.9, 0.999)
    )

    gamma_w = 0.0
    epoch_start = 0

    # ---- Resume ----
    if args.resume is not None and os.path.isfile(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume)
        UVDocnet.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        epoch_start = ckpt["epoch"]
        if epoch_start >= args.ep_gamma_start:
            gamma_w = args.gamma_w

    scheduler = get_scheduler(optimizer, args, epoch_start)

    # ---- Logging dir ----
    os.makedirs(args.logdir, exist_ok=True)

    exp_name = (
        f"UVDOC_bs{args.batch_size}_lr{args.lr}"
        f"_ep{args.n_epochs}_a{args.alpha_w}"
        f"_b{args.beta_w}_g{args.gamma_w}"
    )

    log_file = os.path.join(args.logdir, exp_name + ".txt")
    exp_dir = os.path.join(args.logdir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    with open(log_file, "a") as f:
        f.write(f"\n===== {exp_name} =====\n")

    best_val_mse = float("inf")

    # =========================
    # Training loop
    # =========================
    for epoch in range(epoch_start, args.n_epochs + args.n_epochs_decay):
        print(f"\n===== Epoch {epoch} =====")

        if epoch >= args.ep_gamma_start:
            gamma_w = args.gamma_w

        # ---------- TRAIN ----------
        UVDocnet.train()
        train_mse = 0.0
        count = 0

        pbar = tqdm(
            trainloader,
            desc=f"Train {epoch}",
            total=len(trainloader),
        )

        for imgs_, imgs_unw_, grid2D_, grid3D_ in pbar:
            imgs = imgs_.to(device, non_blocking=True)
            gt_unw = imgs_unw_.to(device, non_blocking=True)
            gt_2d = grid2D_.to(device, non_blocking=True)
            gt_3d = grid3D_.to(device, non_blocking=True)

            pred_2d, pred_3d = UVDocnet(imgs)
            pred_unw = utils.bilinear_unwarping(
                imgs, pred_2d, utils.IMG_SIZE
            )

            loss_recon = criterion_L1(pred_unw, gt_unw)
            loss_2d = criterion_L1(pred_2d, gt_2d)
            loss_3d = criterion_L1(pred_3d, gt_3d)

            loss = (
                args.alpha_w * loss_2d
                + args.beta_w * loss_3d
                + gamma_w * loss_recon
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            mse = criterion_MSE(pred_unw, gt_unw)
            train_mse += mse.detach().item()
            count += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                mse=f"{mse.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        train_mse /= max(1, count)
        lr = update_learning_rate(scheduler, optimizer)
        write_log_file(log_file, train_mse, epoch + 1, lr, "Train")

        gc.collect()
        torch.cuda.empty_cache()

        # ---------- VALID ----------
        UVDocnet.eval()
        val_mse = 0.0

        with torch.no_grad():
            for imgs_, imgs_unw_, _, _ in tqdm(
                valloader,
                desc=f"Val {epoch}",
                total=len(valloader),
            ):
                imgs = imgs_.to(device)
                gt_unw = imgs_unw_.to(device)

                pred_2d, _ = UVDocnet(imgs)
                pred_unw = utils.bilinear_unwarping(
                    imgs, pred_2d, utils.IMG_SIZE
                )

                val_mse += criterion_MSE(pred_unw, gt_unw).item()

        val_mse /= len(valloader)
        write_log_file(log_file, val_mse, epoch + 1, lr, "Val")

        # ---------- SAVE ----------
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": UVDocnet.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                os.path.join(
                    exp_dir,
                    f"best_ep{epoch+1}_val{val_mse:.5f}.pkl",
                ),
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train UVDoc")

    parser.add_argument("--data_path_UVDoc", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--n_epochs_decay", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--alpha_w", type=float, default=5.0)
    parser.add_argument("--beta_w", type=float, default=5.0)
    parser.add_argument("--gamma_w", type=float, default=1.0)
    parser.add_argument("--ep_gamma_start", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="./log/uvdoc")

    parser.add_argument(
        "--appearance_augmentation",
        nargs="*",
        default=["visual", "noise", "color"],
    )
    parser.add_argument(
        "--geometric_augmentationsUVDoc",
        nargs="*",
        default=["rotate"],
    )
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()
    main_worker(args)
