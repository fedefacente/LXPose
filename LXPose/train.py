import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import wandb
from diffdrr.pose import convert
import diffdrr.data
from diffdrr.drr import DRR
from torchvision.transforms import Compose, Lambda
from models import regressor, registration,registration2
from geodesic import GeodesicSE3, DoubleGeodesic
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d,GradientNormalizedCrossCorrelation2d
from pytorch_transformers.optimization import WarmupCosineSchedule
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
import kornia
from kornia.augmentation import AugmentationSequential
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, default=1)
args = parser.parse_args()

# Start a W&B Run with wandb.init
id = args.id
performance = wandb.init(project="Points", name=f'specimen{id}_points', mode = 'offline')

def train(
        model,
        reg_model1,
        optimizer,
        scheduler,
        drr,
        transforms,
        device,
        batch_size,
        n_epochs,
        n_batches_per_epoch,
        parameterization,
        convention,
        output_path,
        subject,
        aug):
    torch.cuda.empty_cache()
    geodesic = GeodesicSE3().to(device)
    n_batches_per_epoch_val = 2
    double = DoubleGeodesic(1020).to(device)
    min_delta = 0.01
    patience = 8
    best_loss = np.inf  # init to infinity
    ncc = MultiscaleNormalizedCrossCorrelation2d([None, 16], [0.5, 0.5]).to(device)
    for epoch in range(n_epochs + 1):
        losses = []
        loss_cnn1 = []
        loss_cnn3 = []
        losses_val = []
        ncc1mean = []
        ncc2mean = []
        model.train()
        reg_model1.train()

        for _ in (itr := tqdm(range(n_batches_per_epoch), leave=False)):

            sid = -750.0
            torch.manual_seed(torch.seed())

            rx = torch.normal(mean=0.0, std=0.2, size=(batch_size, 1)).to(device)
            ry = torch.normal(mean=0.0, std=0.1, size=(batch_size, 1)).to(device)
            rz = torch.normal(mean=0.0, std=0.25, size=(batch_size, 1)).to(device)
            rot = torch.cat([rx, ry, rz], dim=1).to(device)

            tx = torch.normal(mean=0.0, std=70, size=(batch_size, 1)).to(device)
            ty = torch.normal(mean=0, std=90, size=(batch_size, 1)).to(device)
            tz = torch.normal(mean=0.0, std=50, size=(batch_size, 1)).to(device)
            t_gt = torch.cat([tx, ty, tz], dim=1).to(device)

            xyz = torch.cat([tx, ty, tz], dim=1).to(device)
            xyz[:, 1] = xyz[:, 1] + np.float32((0 + sid))

            img = drr(rot, xyz, parameterization=parameterization, convention=convention)
            img = transforms(img)
            img = aug(img)

            r,t = model(img)
            T = t + torch.tensor([np.float32(0), np.float32((0 + sid)), np.float32(0)], dtype=tx.dtype,
                                 device=tx.device)

            pose = convert(rot, xyz, parameterization=parameterization, convention=convention)
            points2d = drr.perspective_projection(pose, subject.fiducials.cuda())

            pose_pred = convert(r, T,parameterization=parameterization, convention=convention)
            points2d_pred = drr.perspective_projection(pose_pred, subject.fiducials.cuda())

            mask1 = (points2d_pred[:, :, 0] >= 0) & (points2d_pred[:, :, 0] <= 256) & (points2d_pred[:, :, 1] >= 0) & (
                    points2d_pred[:, :, 1] <= 256)
            mask2 = (points2d[:, :, 0] >= 0) & (points2d[:, :, 0] <= 256) & (points2d[:, :, 1] >= 0) & (
                    points2d[:, :, 1] <= 256)
            
            mask = mask1 & mask2
            mask2d = mask.unsqueeze(-1).expand_as(points2d_pred)

            points2d_pred[~mask2d] = float('nan')
            points2d[~mask2d] = float('nan')
            mpd1 = ((points2d - points2d_pred).norm(dim=-1).nanmean()) * (0.1940000057220459*1436/256)
            init_pose = drr(r, T, parameterization=parameterization, convention=convention)
            init_pose = transforms(init_pose.sum(dim=1, keepdim=True))
            ncc1 = 1 - ncc(init_pose,img)
          
            input2 = torch.cat((img, init_pose), dim=1)
            delta_r, delta_t = reg_model1(input2)

            T = delta_t + torch.tensor(
                [np.float32(0), np.float32((0 + sid)), np.float32(0)], dtype=tx.dtype,
                device=tx.device)

            pose = convert(rot, xyz, parameterization=parameterization, convention=convention)
            points2d = drr.perspective_projection(pose, subject.fiducials.cuda())

            delta_pose = convert(delta_r, T, parameterization=parameterization, convention=convention)
            pose_pred = pose_pred.compose(delta_pose)

            points2d_pred = drr.perspective_projection(pose_pred, subject.fiducials.cuda())
            mask1 = (points2d_pred[:, :, 0] >= 0) & (points2d_pred[:, :, 0] <= 256) & (points2d_pred[:, :, 1] >= 0) & (
                    points2d_pred[:, :, 1] <= 256)
            mask2 = (points2d[:, :, 0] >= 0) & (points2d[:, :, 0] <= 256) & (points2d[:, :, 1] >= 0) & (
                    points2d[:, :, 1] <= 256)
            
            mask = mask1 & mask2
            mask2d = mask.unsqueeze(-1).expand_as(points2d_pred)

            points2d_pred[~mask2d] = float('nan')
            points2d[~mask2d] = float('nan')
            mpd2 = ((points2d - points2d_pred).norm(dim=-1).nanmean()) * (0.1940000057220459*1436/256)

            final = drr(pose_pred)
            final_pose = transforms(final.sum(dim=1, keepdim=True))
            ncc2 = (1-ncc(final_pose,img))
            del img, final, final_pose
            loss = (ncc2.mean() + 1e-1 * (mpd2.nanmean())) + (ncc1.mean() + 1e-1 * (mpd1.nanmean()))
            optimizer.zero_grad()
            loss.mean().backward()
            adaptive_clip_grad_(model.parameters())
            adaptive_clip_grad_(reg_model1.parameters())
            optimizer.step()
            scheduler.step()
            losses.append(loss.mean().item())
            loss_cnn1.append(mpd1.mean().item())
            loss_cnn2.append(cnn2.mean().item())
            ncc1mean.append(ncc1.mean().item())
            ncc2mean.append(ncc2.mean().item())

            itr.set_description(f"Epoch [{epoch}/{n_epochs}]")
            itr.set_postfix(
                loss=loss.mean().item(),
            )
        losses = torch.tensor(losses)
        loss_cnn1 = torch.tensor(loss_cnn1)
        loss_cnn2 = torch.tensor(loss_cnn2)
        ncc1mean= torch.tensor(ncc1mean)
        ncc2mean= torch.tensor(ncc2mean)

        tqdm.write(f"Epoch {epoch + 1:04d} | Loss {losses.nanmean().item():.4f}")
        performance.log({"Training loss": losses.nanmean().item()})
        performance.log({"mPD cnn1": loss_cnn1.nanmean()})
        performance.log({"mPD cnn2": loss_cnn2.nanmean()})
        performance.log({"ncc cnn1": ncc1mean.nanmean()})
        performance.log({"ncc cnn2": ncc2mean.nanmean()})
        if (epoch + 1) % 50 == 0:
            model.eval()
            with (((torch.no_grad()))):
                for _ in (tqdm(range(n_batches_per_epoch_val), leave=False)):
                    sid = -750.0
                    torch.manual_seed(0)

                    rx = torch.normal(mean=0.0, std=0.2, size=(batch_size, 1)).to(device)
                    ry = torch.normal(mean=0.0, std=0.1, size=(batch_size, 1)).to(device)
                    rz = torch.normal(mean=0.0, std=0.25, size=(batch_size, 1)).to(device)

                    tx = torch.normal(mean=0.0, std=70, size=(batch_size, 1)).to(device)
                    ty = torch.normal(mean=0, std=90, size=(batch_size, 1)).to(device)
                    tz = torch.normal(mean=0.0, std=50, size=(batch_size, 1)).to(device)

                    rot = torch.cat([rx, ry, rz], dim=1).to(device)
                    xyz = torch.cat([tx, ty, tz], dim=1).to(device)
                    xyz[:, 1] = xyz[:, 1] + np.float32((0 + sid))

                    img = drr(rot, xyz, parameterization=parameterization, convention=convention)
                    img = transforms(img)
                    img = aug(img)

                    r, t = model(img)
                    T = t + torch.tensor([np.float32(0), np.float32((0 + sid)), np.float32(0)], dtype=tx.dtype,
                                         device=tx.device)

                    pose = convert(rot, xyz, parameterization=parameterization, convention=convention)
                    points2d = drr.perspective_projection(pose, subject.fiducials.cuda())

                    pose_pred = convert(r, T, parameterization=parameterization, convention=convention)
                    points2d_pred = drr.perspective_projection(pose_pred, subject.fiducials.cuda())

                    mask1 = (points2d_pred[:, :, 0] >= 0) & (points2d_pred[:, :, 0] <= 256) & (
                                points2d_pred[:, :, 1] >= 0) & (
                                    points2d_pred[:, :, 1] <= 256)
                    mask2 = (points2d[:, :, 0] >= 0) & (points2d[:, :, 0] <= 256) & (points2d[:, :, 1] >= 0) & (
                            points2d[:, :, 1] <= 256)

                    mask = mask1 & mask2
                    mask2d = mask.unsqueeze(-1).expand_as(points2d_pred)

                    points2d_pred[~mask2d] = float('nan')
                    points2d[~mask2d] = float('nan')
                    mpd1 = ((points2d - points2d_pred).norm(dim=-1).nanmean()) * (0.1940000057220459 * 1436 / 256)
                    
                    init_pose = drr(r, T, parameterization=parameterization, convention=convention)
                    init_pose = transforms(init_pose.sum(dim=1, keepdim=True))
                    ncc1 = 1 - ncc(init_pose, img)
                    input2 = torch.cat((img, init_pose), dim=1)

                    delta_r, delta_t = reg_model1(input2)

                    T = delta_t + torch.tensor(
                        [np.float32(0), np.float32((0 + sid)), np.float32(0)], dtype=tx.dtype,
                        device=tx.device)
                    pose = convert(rot, xyz, parameterization=parameterization, convention=convention)
                    points2d = drr.perspective_projection(pose, subject.fiducials.cuda())

                    delta_pose = convert(delta_r, T, parameterization=parameterization, convention=convention)
                    pose_pred = pose_pred.compose(delta_pose)

                    points2d_pred = drr.perspective_projection(pose_pred, subject.fiducials.cuda())
                    mask1 = (points2d_pred[:, :, 0] >= 0) & (points2d_pred[:, :, 0] <= 256) & (
                                points2d_pred[:, :, 1] >= 0) & (
                                    points2d_pred[:, :, 1] <= 256)
                    mask2 = (points2d[:, :, 0] >= 0) & (points2d[:, :, 0] <= 256) & (points2d[:, :, 1] >= 0) & (
                            points2d[:, :, 1] <= 256)

                    mask = mask1 & mask2
                    mask2d = mask.unsqueeze(-1).expand_as(points2d_pred)

                    points2d_pred[~mask2d] = float('nan')
                    points2d[~mask2d] = float('nan')

                    mpd2 = ((points2d - points2d_pred).norm(dim=-1).nanmean()) * (0.1940000057220459 * 1436 / 256)

                    final = drr(pose_pred)
                    final_pose = transforms(final.sum(dim=1, keepdim=True))
                    ncc2 = (1 - ncc(final_pose, img))
                    del img, final, final_pose
                    loss = (ncc2.mean() + 1e-1 * (mpd2.nanmean())) + (ncc1.mean() + 1e-1 * (mpd1.nanmean()))
                    losses_val.append(loss.mean().item())

                loss = torch.tensor(losses_val).nanmean().item()
                tqdm.write(f"Epoch {epoch + 1:04d} | Validation loss {loss:.4f}")
                performance.log({"Validation loss": loss})
                torch.save(
                    {
                        "model_state_dict1": model.state_dict(),
                        "model_state_dict2": reg_model1.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    os.path.join(output_path, f'model_{epoch}.pth'),
                )
                if loss < best_loss - min_delta:
                    counter = 0
                    best_loss = loss
                    torch.save(
                        {
                            "model_state_dict1": model.state_dict(),
                            "model_state_dict2": reg_model1.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "epoch": epoch,
                        },
                        os.path.join(output_path, 'best1.pth'),
                    )
                else:
                    counter += 1
                    if counter >= patience:
                        print("Early stopping.")
                        break

def main(
    parameterization="axis_angle",
    convention=None,
    lr=1e-3,
    batch_size=16,
    n_epochs=1000,
    n_batches_per_epoch=100,
):

    device = torch.device("cuda")
    csv_file_path = f"/lustre/fswork/projects/rech/pin/uur34ii/data/6_DOF_estimation/fiducials/specimen_{id}.csv"
    df = pd.read_csv(csv_file_path)
    fiducials = np.array(df.values)  # Extract as NumPy array
    fiducials = torch.tensor([fiducials]).to(torch.float)

    subject = diffdrr.data.read(f"/lustre/fswork/projects/rech/pin/uur34ii/data/6_DOF_estimation/volumes_RAS/specimen_{id}.nii.gz",
                                orientation="PA", bone_attenuation_multiplier=2, fiducials=fiducials)
    drr = DRR(
        subject,  # A torchio.Subject object storing the CT volume, origin, and voxel spacing
        sdd=1020,  # Source-to-detector distance (i.e., the C-arm's focal length)
        height=256,  # Height of the DRR (if width is not seperately provided, the generated image is square)
        delx=(0.1940000057220459*1436/256),  # Pixel spacing (in mm)
        renderer="trilinear",
        reverse_x_axis=True
    ).to(device)

    model = regressor()
    reg_model1 = registration()
    
    model = model.to(device)
    reg_model1 = reg_model1.to(device)
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': lr},
        {'params': reg_model1.parameters(), 'lr': lr}])
  

    scheduler = WarmupCosineSchedule(
        optimizer,
        5 * n_batches_per_epoch,
        n_epochs * n_batches_per_epoch - 5 * n_batches_per_epoch,
    )

    output_path = f"/lustre/fswork/projects/rech/pin/uur34ii/model/RigidReg/specimen_{id}"
    os.makedirs(output_path, exist_ok=True)

    transforms = Compose(
        [
            Lambda(lambda x: (((x - x.min()) / (x.max() - x.min()+ 1e-6)))),
        ]
    )


    aug = AugmentationSequential(
        kornia.augmentation.RandomSaltAndPepperNoise(amount=(0.005, 0.01), salt_vs_pepper=(0.4, 0.6), p=0.5,same_on_batch=False, keepdim=True),
        kornia.augmentation.RandomGamma(gamma=(0.6, 1.8), gain=(1.0, 1.0), same_on_batch=False, p=0.5, keepdim=True),
        kornia.augmentation.RandomGaussianNoise(mean=0.0, std=0.03, same_on_batch=False, p=0.5, keepdim=True),
        kornia.augmentation.RandomSharpness(p=0.5, keepdim=True, same_on_batch=False),
        kornia.augmentation.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1), p=0.5, keepdim=True,
                                               same_on_batch=False),
        kornia.augmentation.RandomPlasmaContrast(roughness=(0.1, 0.5),p=1, keepdim=True, same_on_batch=False),
        kornia.augmentation.RandomPlasmaBrightness(roughness=(0.1, 0.5), intensity=(-1, 1),p=1, keepdim=True, same_on_batch=False),

        data_keys=["input"],
        same_on_batch=None,
    )

    train(
        model,
        reg_model1,
        optimizer,
        scheduler,
        drr,
        transforms,
        device,
        batch_size,
        n_epochs,
        n_batches_per_epoch,
        parameterization,
        convention,
        output_path,
        subject,
        aug)

if __name__ == "__main__":
    main()
