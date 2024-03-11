# import argparse
import logging
import os
from pathlib import Path
import time
from sklearn.metrics import roc_auc_score

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from d_UNet.model_improved import DDCA_UNet, DU_Net
from utils.data_loading import FlowDataset
from utils.dice_score import dice_coeff
# from utils.utils import plot_img_and_mask

dir_img = Path('./data_test/imgs_add')
dir_mask = Path('./data_test/masks/')
dir_noise = Path('./data_test/noise_map/')
dir_combine = Path('./data_test/combine/')
dir_lap = Path('./data_test/lap_feature')
dir_model = Path('./checkpoints/save/checkpoint_epoch419_denoise_8_23_bs6.pth')
dir_ground = Path('./data_train/masks_total')
no_save = False
classes = 1


def mask_to_image(mask: np.ndarray, shape, classes):
    if classes == 1:
        if shape[0] == 3:
            print(3)
            return Image.fromarray((mask * 255).astype(np.uint8), mode='RGB')
        else:
            return Image.fromarray((mask * 255).astype(np.uint8))
    elif classes ==2:
        return Image.fromarray((np.argmax(mask, axis=0) * 255).astype(np.uint8))


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = FlowDataset.preprocess(full_img, scale_factor, is_mask=False, is_test=True)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output, lap, noise_map, combine = net(img)
        # output = net(img)

        #if net.n_classes > 1:
        if classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        # tf = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((full_img.size[1], full_img.size[0])),
        #     transforms.ToTensor()
        # ])
        #
        # full_mask = tf(probs.cpu()).squeeze()
        full_mask = probs.cpu().squeeze()

    #if net.n_classes == 1:
    if classes == 1:
        return (full_mask > out_threshold).numpy(), lap.cpu().squeeze().numpy(), noise_map.cpu().squeeze().numpy(), \
               combine.squeeze().permute(1, 2, 0).cpu().numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), classes).permute(2, 0, 1).numpy(), lap.cpu().squeeze().numpy(),\
            noise_map.squeeze().permute(1, 2, 0).cpu().numpy()


def evaluate(net, dataloader, device):
    net.eval()
    MAE = 0
    count = 0
    P = 0
    R = 0
    F1 = 0
    Time = 0
    pixel_acc = 0
    IoU = 0
    auc = 0
    dice = 0
    epsilon = 1e-6

    # iterate over the validation set
    for batch in dataloader: #tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        #mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            start_t = time.time()
            mask_pred, _, _, _ = net(image)
            # mask_pred = net(image)
            end_t = time.time()
            Time += end_t-start_t

            if classes > 1:
                mask_pred = mask_pred.argmax(dim=1)
            else:
                mask_pred = torch.sigmoid(mask_pred.squeeze(1))
                mask_pred = torch.round(mask_pred)

            for bs in range(mask_true.shape[0]):
                predict = mask_pred[bs]
                true = mask_true[bs]
                # # Dice
                # dice += dice_coeff(predict, true.float())
                # AUC
                pred_temp = predict.view(-1,1)
                true_temp = true.view(-1,1)
                auc += roc_auc_score(true_temp.cpu(), pred_temp.cpu())
                # Pixel_Accuracy
                pixel_cor = torch.eq(predict, true).sum().item()
                pixel_total = true.numel()
                pixel_acc += pixel_cor/pixel_total
                # IoU
                n = (predict * true).sum().item()
                u1 = true.sum().item()
                u2 = predict.sum().item()
                IoU += n / (u1 + u2 - n)
                # MAE
                error = torch.abs(predict-true).sum().item()
                pixel_total = true.numel()
                MAE += error/pixel_total
                # F1
                TP = (predict * true).sum().item()
                FP = (predict * (1-true)).sum().item()
                FN = ((1-predict) * true).sum().item()
                P_temp = (TP+epsilon)/(TP+FP+epsilon)
                R_temp = (TP+epsilon)/(TP+FN+epsilon)
                P += P_temp
                R += R_temp
                F1 += (2*P_temp*R_temp+epsilon)/(P_temp+R_temp+epsilon)

                count += 1

    return R/count ,P/count, F1/count,MAE/count, Time/count, pixel_acc/count, IoU/count, auc/count


if __name__ == '__main__':
    # args = get_args()
    # in_files = args.input
    # out_files = get_output_filenames(args)

    in_files=dir_img
    out_files=dir_mask
    trainedmodel=dir_model

    # net = DC_Bo_Transformer(img_ch=3, output_ch=1)
    net = DDCA_UNet(img_ch=3, output_ch=1, feature_show=True)
    # net = DU_Net(in_ch=3,out_ch=1)

    device = torch.device('cuda')#'cuda' if torch.cuda.is_available() else
    # logging.info(f'Loading model {trainedmodel}')
    # logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(trainedmodel, map_location=device))

    print(net.de_noise.kernel)
    print('Model loaded!')

    dataset = FlowDataset(str(dir_img), str(dir_ground))
    test_loader = DataLoader(dataset, shuffle=False, drop_last=True)
    #R, P, F1, MAE, Time, PA, IoU, AUC = evaluate(net, test_loader, device)
    #print('Test R:{} P:{} F1:{} MAE:{} Time:{} PA:{} IoU:{} AUC:{}'.format(R, P, F1, MAE, Time, PA, IoU, AUC))

    imglist = os.listdir(in_files)
    print('Mask saved to {}'.format(dir_mask))
    for filename in imglist:
        print('Predicting image {} '.format(filename))
        img = Image.open(os.path.join(in_files,filename))

        mask, lap, noise_map, combine = predict_img(net=net,
                           full_img=img,
                           # scale_factor=args.scale,
                           # out_threshold=args.mask_threshold,
                           device=device)  #

        # print("lapshape=",lap.shape)
        if not no_save:
            result = mask_to_image(mask, mask.shape, classes=classes)
            result.save(os.path.join(dir_mask, filename.replace('.jpg', '.png')))
            result = mask_to_image(noise_map, noise_map.shape, classes=classes)
            result.save(os.path.join(dir_noise, filename.replace('.jpg', '.png')))
            result = mask_to_image(combine, combine.shape, classes=classes)
            result.save(os.path.join(dir_combine, filename.replace('.jpg', '.png')))
            result = mask_to_image(lap[0], lap[0].shape, classes=classes)
            result.save(os.path.join(dir_lap, filename.replace('.jpg', '_r.png')))
            # result = mask_to_image(lap[1], lap.shape, classes=classes)
            # result.save(os.path.join(dir_lap, filename.replace('.jpg', '_g.png')))
            # result = mask_to_image(lap[2], lap.shape, classes=classes)
            # result.save(os.path.join(dir_lap, filename.replace('.jpg', '_b.png')))


        # if viz:
        #     logging.info(f'Visualizing results for image {filename}, close to continue...')
        #     plot_img_and_mask(img, mask)
