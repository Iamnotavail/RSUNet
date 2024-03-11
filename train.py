import logging
import sys
from pathlib import Path
import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split

from utils.data_loading import FlowDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from d_UNet.model_improved import DDCA_UNet, NoLap_DDCA_UNet

classes = 1
use_wandb = False
dir_img_train = './data_train/imgs_total/'
dir_mask = './data_train/masks_total/'
dir_img_val = './data_train/imgs_val'
dir_checkpoint = Path('./checkpoints')

def train_net(net,
              device,
              epochs: int = 500,
              batch_size: int = 6,
              learning_rate: float = 0.001,
              save_checkpoint: bool = True,
              amp: bool = False,
              weight_decay=0,
              num_workers=4,
              ):
    # 1. Create dataset
    train_set = FlowDataset(dir_img_train, dir_mask)
    train_set.is_aug = True
    val_set = FlowDataset(dir_img_val, dir_mask)
    val_set.is_aug = False

    # 2. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True,**loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_set)}
        Validation size: {len(val_set)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
        Classes:         {classes}
        Weight_Decay:    {weight_decay}
    ''')
    if use_wandb:
        wandb.config.epoch = epochs
        wandb.config.batch_size = batch_size
        wandb.config.learning_rate = learning_rate
        wandb.config.train_size = len(train_set)
        wandb.config.validation_size = len(val_set)
        wandb.config.classes = classes
        wandb.config.weight_decay = weight_decay

    # 3. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate,
                           betas=(0.9, 0.999), weight_decay=weight_decay, eps=1e-8)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[350], gamma=0.3)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.02, 0.98]).to(device= device, dtype=torch.float32), reduction='mean')
    criterion_2 = nn.BCEWithLogitsLoss()
    global_step = 0
    best_p = 0
    best_auc = 0

    # 4. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        batch_step=0
        for batch in train_loader:
            images = batch['image']
            true_masks = batch['mask']

            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=amp):
                masks_pred = net(images)

                loss = criterion_2(masks_pred.squeeze(1), true_masks.float()) \
                       + dice_loss(torch.sigmoid(masks_pred.squeeze(1)).float(),
                                   true_masks.float(),
                                   multiclass=False)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()
            batch_step += 1


        PA, IoU, auc = evaluate(net, val_loader, device, classes=classes)
        print(epoch+1, optimizer.state_dict()['param_groups'][0]['lr'], epoch_loss/batch_step,PA,IoU,auc)
        if use_wandb:
            wandb.log({
                "epoch": epoch+1,
                "learning_rate": optimizer.state_dict()['param_groups'][0]['lr'],
                "loss": epoch_loss/batch_step,
                "PA": PA,
                "IoU": IoU,
                "AUC": auc
            })

        scheduler.step()

        if save_checkpoint:
            if IoU > best_p and IoU > 0.2:
                best_p = IoU
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0')
    logging.info(f'Using device {device}')

    wandb.init(project="improve_model_1", entity="tingb")
    wandb.config.model = 'NoLap_DDCA_UNet'

    #net = DDCA_UNet(img_ch=3, output_ch=1)
    net = NoLap_DDCA_UNet(img_ch=3, output_ch=1)
    # net = DC_MHSA(img_ch=3, output_ch=1)
    #net = DC_Bo_Transformer(img_ch=3, output_ch=1)
    #net.load_from(weights=np.load('./Vit_weight/ViT-B_16.npz'))
    #net = UNet_Transformer_test(in_ch=3, out_ch=1)
    #net = UNet_Transformer_0(in_ch=3, out_ch=1)

    net.to(device=device)
    if is_pretrain:
        trainedmodel = dir_model
        net.load_state_dict(torch.load(trainedmodel, map_location=device))
    try:
        train_net(net=net,device=device)
                  # epochs=args.epochs,
                  # batch_size=args.batch_size,
                  # learning_rate=args.lr,
                  # device=device,
                  # img_scale=args.scale,
                  # val_percent=args.val / 100,
                  # amp=args.amp)
    except KeyboardInterrupt:
        # torch.save(net.state_dict(), 'INTERRUPTED.pth')
        # logging.info('Saved interrupt')
        sys.exit(0)

# 改变学习率
#     for choice in (4, 5):
#         f_t = open(dir_text, 'a')
#         if choice == 1:
#             print("---------CE_loss---------")
#             f_t.write('---------CE_loss---------\n')
#         elif choice == 2:
#             print("---------Dice_loss---------")
#             f_t.write('---------Dice_loss---------\n')
#         elif choice == 3:
#             print("---------混合loss_CE---------")
#             f_t.write('---------混合loss_CE---------\n')
#         elif choice == 4:
#             print("---------BCE_loss---------")
#             f_t.write('---------BCE_loss---------\n')
#         elif choice == 5:
#             print("---------混合loss_BCE---------")
#             f_t.write('---------混合loss_BCE---------\n')
#         f_t.close()
#     for lr in (0.005, 0.002):
#         print("lr=", lr)
#         logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
#         device = torch.device('cuda:2')  # 'cuda' if torch.cuda.is_available() else 'cpu'
#         logging.info(f'Using device {device}')
#
#         run = wandb.init(project="unet-compare1", entity="tingb", reinit=True)
#         wandb.config.model = 'unet'
#
#         # Change here to adapt to your data
#         # n_channels=3 for RGB images
#         # n_classes is the number of probabilities you want to get per pixel
#         net = U_Net(in_ch=3, out_ch=classes)
#         # net = SegNet(input_channels=3, output_channels=2)
#         # net = ResNetUNet(n_classes=2)
#         # net = AttU_Net(img_ch=3,output_ch=2)
#         # net = ResAttU_Net(img_ch=3,output_ch=2)
#         # net = DCRes_MaxPoolAtt_UNet(img_ch=3,output_ch=classes)
#
#         # logging.info(f'Network:\n'
#         #              f'\t{net.n_channels} input channels\n'
#         #              f'\t{net.n_classes} output channels (classes)\n'
#         #              f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
#
#         net.to(device=device)
#         if is_pretrain:
#             trainedmodel = dir_model
#             net.load_state_dict(torch.load(trainedmodel, map_location=device))
#         try:
#             train_net(net=net, device=device, learning_rate=lr,
#                       )
#             # epochs=args.epochs,
#             # batch_size=args.batch_size,
#             # ,
#             # device=device,
#             # img_scale=args.scale,
#             # val_percent=args.val / 100,
#             # amp=args.amp)
#         except KeyboardInterrupt:
#             # torch.save(net.state_dict(), 'INTERRUPTED.pth')
#             # logging.info('Saved interrupt')
#             sys.exit(0)
#         run.finish()

#循环训练
    # for repeat in (5, -1):
    #     if repeat == -1:
    #         break
    #     logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    #     device = torch.device('cuda:2')#'cuda' if torch.cuda.is_available() else 'cpu'
    #     logging.info(f'Using device {device}')
    #
    #     run = wandb.init(project="model-compare1", entity="tingb", reinit=True)
    #
    #     if repeat == 0:
    #         net = AttU_Net(img_ch=3, output_ch=classes)
    #         dir_checkpoint = Path('./checkpoints/attunet')
    #         print("------attunet------")
    #         wandb.config.model = 'att_unet'
    #     elif repeat == 1:
    #         net = ResAttU_Net(img_ch=3, output_ch=2)
    #         dir_checkpoint = Path('./checkpoints/resattunet')
    #         print("------resattunet------")
    #     elif repeat == 2:
    #         net = U_Net(in_ch=3, out_ch=classes)
    #         dir_checkpoint = Path('./checkpoints/unet')
    #         print("------unet------")
    #         wandb.config.model = 'unet'
    #     elif repeat == 3:
    #         net = SegNet(input_channels=3, output_channels=classes)
    #         dir_checkpoint = Path('./checkpoints/segnet')
    #         print("------segnet------")
    #     elif repeat == 4:
    #         net = ResNetUNet(n_classes=classes)
    #         dir_checkpoint = Path('./checkpoints/resnetunet')
    #         print("------resnetunet------")
    #     elif repeat == 5:
    #         net = DC_UNet(img_ch=3, output_ch=classes)
    #         dir_checkpoint = Path('./checkpoints/test_temp')
    #         print("------DC_UNet------")
    #         wandb.config.model = 'DC_UNet_2'
    #     elif repeat == 6:
    #         net = ResBlock_UNet(img_ch=3, output_ch=classes)
    #         dir_checkpoint = Path('./checkpoints/improveunet')
    #         print("------ResBlock_UNet------")
    #         wandb.config.model = 'ResBlock_UNet'
    #     elif repeat == 7:
    #         net = DCRes_UNet(img_ch=3, output_ch=classes)
    #         dir_checkpoint = Path('./checkpoints/test_temp')
    #         print("------DCRes_UNet------")
    #         wandb.config.model = 'DCRes_UNet'
    #     elif repeat == 8:
    #         net = DCRes_MaxPoolAtt_UNet(img_ch=3, output_ch=classes)
    #         dir_checkpoint = Path('./checkpoints/DCresmaxpoolattunet')
    #         print("------DCRes_MaxPoolAtt_UNet------")
    #         wandb.config.model = 'DCRes_MaxPoolAtt_UNet'
    #     elif repeat == 9:
    #         net = DCRes_SEAtt_UNet(img_ch=3, output_ch=2)
    #         dir_checkpoint = Path('./checkpoints/DCresseattunet')
    #         print("------DCRes_SEAtt_UNet------")
    #     elif repeat == 10:
    #         a=1
    #     elif repeat == 11:
    #         net = DCRes_MaxPoolAtt_2_UNet(img_ch=3, output_ch=2)
    #         dir_checkpoint = Path('./checkpoints/ablation/att_2')
    #         print("------DCRes_MaxPoolAtt_2_UNet------")
    #     elif repeat == 12:
    #         net = DCRes_MaxPoolAtt_across_UNet(img_ch=3, output_ch=classes)
    #         dir_checkpoint = Path('./checkpoints/ablation/att_cross')
    #         print("------DCRes_MaxPoolAtt_across_UNet------")
    #     elif repeat == 13:
    #         net = DCRes_AvgPoolAtt_UNet(img_ch=3, output_ch=classes)
    #         dir_checkpoint = Path('./checkpoints/ablation/att_avg')
    #         print("------DCRes_AvgPoolAtt_UNet------")
    #     elif repeat == 14:
    #         net = DCRes_UNet(img_ch=3, output_ch=classes)
    #         dir_checkpoint = Path('./checkpoints/ablation/dcres')
    #         print("------DCRes_UNet------")
    #     elif repeat == 15:
    #         net = MaxPoolAtt_UNet(img_ch=3, output_ch=classes)
    #         dir_checkpoint = Path('./checkpoints/ablation/maxpool')
    #         print("------MaxPoolAtt_UNet------")
    #         wandb.config.model = 'MaxPoolAtt_UNet'
    #     elif repeat == 16:
    #         net = test_sk_UNet(img_ch=3, output_ch=classes)
    #         dir_checkpoint = Path('./checkpoints/test_temp')
    #         print("------test_sk_UNet------")
    #         wandb.config.model = 'test_sk_UNet'
    #
    #
    #     net.to(device=device)
    #     if is_pretrain:
    #         trainedmodel = dir_model
    #         net.load_state_dict(torch.load(trainedmodel, map_location=device))
    #     try:
    #         train_net(net=net,device=device)
    #                   # epochs=args.epochs,
    #                   # batch_size=args.batch_size,
    #                   # learning_rate=args.lr,
    #                   # device=device,
    #                   # img_scale=args.scale,
    #                   # val_percent=args.val / 100,
    #                   # amp=args.amp)
    #     except KeyboardInterrupt:
    #         # torch.save(net.state_dict(), 'INTERRUPTED.pth')
    #         # logging.info('Saved interrupt')
    #         sys.exit(0)
    #
    #     run.finish()

# 样本数量
#     for tsz in (50, 100, 150, 200, 250):
#         print("train_size=", tsz)
#         logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
#         device = torch.device('cuda:1')  # 'cuda' if torch.cuda.is_available() else 'cpu'
#         logging.info(f'Using device {device}')
#
#         # Change here to adapt to your data
#         # n_channels=3 for RGB images
#         # n_classes is the number of probabilities you want to get per pixel
#         # net = U_Net(in_ch=3, out_ch=2)
#         # net = SegNet(input_channels=3, output_channels=2)
#         # net = ResNetUNet(n_classes=2)
#         # net = AttU_Net(img_ch=3,output_ch=2)
#         # net = ResAttU_Net(img_ch=3,output_ch=2)
#         net = DCRes_MaxPoolAtt_UNet(img_ch=3,output_ch=classes)
#
#         # logging.info(f'Network:\n'
#         #              f'\t{net.n_channels} input channels\n'
#         #              f'\t{net.n_classes} output channels (classes)\n'
#         #              f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
#
#         net.to(device=device)
#         if is_pretrain:
#             trainedmodel = dir_model
#             net.load_state_dict(torch.load(trainedmodel, map_location=device))
#         try:
#             train_net(net=net, device=device, train_size=tsz,
#                       save_checkpoint=True)
#             # epochs=args.epochs,
#             # batch_size=args.batch_size,
#             # ,
#             # device=device,
#             # img_scale=args.scale,
#             # val_percent=args.val / 100,
#             # amp=args.amp)
#         except KeyboardInterrupt:
#             # torch.save(net.state_dict(), 'INTERRUPTED.pth')
#             # logging.info('Saved interrupt')
#             sys.exit(0)
