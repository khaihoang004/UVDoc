import argparse
import gc
import os

import torch

import data_UVDoc
import model
import utils

train_mse = 0.0
losscount = 0
gamma_w = 0.0


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

    # Eval dataset: same UVDoc but NO augmentation
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
        lr_l = 1.0 - max(0, epoch + epoch_start - args.n_epochs) / float(args.n_epochs_decay + 1)
        return lr_l

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)


def update_learning_rate(scheduler, optimizer):
    old_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    lr = optimizer.param_groups[0]["lr"]
    print("learning rate update from %.7f -> %.7f" % (old_lr, lr))
    return lr


def write_log_file(log_file_name, loss, epoch, lrate, phase):
    with open(log_file_name, "a") as f:
        f.write("\n{} LRate: {} Epoch: {} MSE: {:.5f} ".format(phase, lrate, epoch, loss))


def main_worker(args):
    trainloader, valloader = setup_data(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    UVDocnet = model.UVDocnet(num_filter=32, kernel_size=5)
    UVDocnet.to(device)

    # Losses
    criterionL1 = torch.nn.L1Loss()
    criterionMSE = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(UVDocnet.parameters(), lr=args.lr, betas=(0.9, 0.999))

    global gamma_w
    epoch_start = 0

    if args.resume is not None and os.path.isfile(args.resume):
        print(f"Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume)
        UVDocnet.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        epoch_start = checkpoint["epoch"]
        if epoch_start >= args.ep_gamma_start:
            gamma_w = args.gamma_w

    scheduler = get_scheduler(optimizer, args, epoch_start)

    # Logging
    os.makedirs(args.logdir, exist_ok=True)

    experiment_name = (
        "UVDOC_"
        + "bs"
        + str(args.batch_size)
        + "_lr"
        + str(args.lr)
        + "_nep"
        + str(args.n_epochs)
        + "_alpha"
        + str(args.alpha_w)
        + "_beta"
        + str(args.beta_w)
        + "_gamma"
        + str(args.gamma_w)
    )

    log_file_name = os.path.join(args.logdir, experiment_name + ".txt")
    with open(log_file_name, "a") as f:
        f.write("\n---------------  " + experiment_name + "  ---------------\n")

    exp_log_dir = os.path.join(args.logdir, experiment_name)
    os.makedirs(exp_log_dir, exist_ok=True)

    global losscount, train_mse

    # ==========================
    # Training loop
    # ==========================
    for epoch in range(epoch_start, args.n_epochs + args.n_epochs_decay + 1):
        print(f"\n----- Epoch {epoch} -----")

        if epoch >= args.ep_gamma_start:
            gamma_w = args.gamma_w
            print("gamma_w activated:", gamma_w)

        train_mse = 0.0
        losscount = 0
        best_val_mse = float("inf")

        UVDocnet.train()

        for imgs_, imgs_unwarped_, grid2D_, grid3D_ in trainloader:
            imgs = imgs_.to(device, non_blocking=True)
            unwarped_GT = imgs_unwarped_.to(device, non_blocking=True)
            grid2D_GT = grid2D_.to(device, non_blocking=True)
            grid3D_GT = grid3D_.to(device, non_blocking=True)

            grid2D_pred, grid3D_pred = UVDocnet(imgs)
            unwarped_pred = utils.bilinear_unwarping(imgs, grid2D_pred, utils.IMG_SIZE)

            optimizer.zero_grad(set_to_none=True)

            recon_loss = criterionL1(unwarped_pred, unwarped_GT)
            loss_grid2D = criterionL1(grid2D_pred, grid2D_GT)
            loss_grid3D = criterionL1(grid3D_pred, grid3D_GT)

            netLoss = args.alpha_w * loss_grid2D + args.beta_w * loss_grid3D + gamma_w * recon_loss
            netLoss.backward()
            optimizer.step()

            tmp_mse = criterionMSE(unwarped_pred, unwarped_GT)
            train_mse += float(tmp_mse)
            losscount += 1

            gc.collect()

        train_mse /= max(1, losscount)
        curr_lr = update_learning_rate(scheduler, optimizer)
        write_log_file(log_file_name, train_mse, epoch + 1, curr_lr, "Train")

        # Validation
        UVDocnet.eval()
        with torch.no_grad():
            mse_loss_val = 0.0
            for imgs_, imgs_unwarped_, _, _ in valloader:
                imgs = imgs_.to(device)
                unwarped_GT = imgs_unwarped_.to(device)

                grid2D_pred, _ = UVDocnet(imgs)
                unwarped_pred = utils.bilinear_unwarping(imgs, grid2D_pred, utils.IMG_SIZE)

                loss_img_val = criterionMSE(unwarped_pred, unwarped_GT)
                mse_loss_val += float(loss_img_val)

            val_mse = mse_loss_val / len(valloader)
            write_log_file(log_file_name, val_mse, epoch + 1, curr_lr, "Val")

        # Save best
        if val_mse < best_val_mse or epoch == args.n_epochs + args.n_epochs_decay:
            best_val_mse = val_mse
            state = {
                "epoch": epoch + 1,
                "model_state": UVDocnet.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            model_path = os.path.join(
                exp_log_dir,
                f"ep_{epoch + 1}_{val_mse:.5f}_{train_mse:.5f}_best_model.pkl",
            )
            torch.save(state, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UVDoc only")

    parser.add_argument("--data_path_UVDoc", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--n_epochs_decay", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--alpha_w", type=float, default=5.0)
    parser.add_argument("--beta_w", type=float, default=5.0)
    parser.add_argument("--gamma_w", type=float, default=1.0)
    parser.add_argument("--ep_gamma_start", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="./log/uvdoc")

    parser.add_argument(
        "-a",
        "--appearance_augmentation",
        nargs="*",
        default=["visual", "noise", "color"],
        choices=["shadow", "blur", "visual", "noise", "color"],
    )
    parser.add_argument(
        "-gUVDoc",
        "--geometric_augmentationsUVDoc",
        nargs="*",
        default=["rotate"],
        choices=["rotate", "flip", "perspective"],
    )
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()
    main_worker(args)
