import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils.dice_score import multiclass_dice_coeff, dice_coeff


def evaluate(net, dataloader, device, classes):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    pixel_cor = 0
    pixel_total = 0
    pixel_acc = 0
    IoU = 0
    auc = 0
    count = 0

    # iterate over the validation set
    for batch in dataloader: #tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        #mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            # if net.n_classes == 1:
            #     mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            #     # compute the Dice score
            #     dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            # else:
            #     mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            #     # compute the Dice score, ignoring background
            #     dice_score += multiclass_dice_coeff(mask_pred, mask_true,
            #                                         reduce_batch_first=False)
            #mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            if classes > 1:
                mask_pred = mask_pred.argmax(dim=1)
            else:
                mask_pred = torch.sigmoid(mask_pred.squeeze(1))
                mask_pred = torch.round(mask_pred)

            if mask_true.shape != mask_pred.shape:
                print("网络输出与ground truth维数不同")
            # compute the Dice score, ignoring background
            for bs in range(mask_true.shape[0]):
                predict = mask_pred[bs]
                true = mask_true[bs]
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

                count += 1


            # mask_true = mask_true.cpu()
            # mask_pred = mask_pred.cpu()
            # for i in range(mask_true.shape[0]):
            #     for j in range(mask_true.shape[1]):
            #         for k in range(mask_true.shape[2]):
            #             if mask_true[i][j][k] == 1 or mask_pred[i][j][k] == 1:
            #                 u+=1
            #             if mask_true[i][j][k] == 1 and mask_pred[i][j][k] == 1:
            #                 n+=1
            # dice_score += dice_coeff(mask_pred, mask_true,
            #                                     reduce_batch_first=False)

    net.train()
    return pixel_acc/count, IoU/count, auc/count
