import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import random
import numpy

import PIL.Image
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms


def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
    A = numpy.matrix(matrix, dtype=numpy.float)
    B = numpy.array(pb).reshape(8)
    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    return numpy.array(res).reshape(8)

class FlowDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        self.is_aug = False

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask, is_test = False):
        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # pil_img = pil_img.resize((newW, newH))
        # if not is_test:
        #     pil_img = pil_img.resize((256, 256))
        # img_ndarray = np.asarray(pil_img)

        if not is_mask:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.5182, 0.5361, 0.4892), (0.1611, 0.1586, 0.1784)),
            ])
            pil_img = transform_train(pil_img)
        else:
            pil_img = np.asarray(pil_img)
        #print(pil_img)
        # if img_ndarray.ndim == 2 and not is_mask:
        #     img_ndarray = img_ndarray[np.newaxis, ...]
        # elif not is_mask:
        #     img_ndarray = img_ndarray.transpose((2, 0, 1))
        #
        # if not is_mask:
        #     img_ndarray = img_ndarray / 255

        return pil_img

    @classmethod
    def augment(cls, image, mask):
        # 以p概率旋转
        p = random.random()
        if p < 0.4:
            r = random.randint(1, 3)
            image = image.rotate(90*r)
            mask = mask.rotate(90*r)
        elif p < 0.6:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        elif p < 0.8:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        # # 以p概率旋转
        # p = random.random()
        # if p < 0.3:
        #     r = random.randint(-5, 5)
        #     image = image.rotate(r)
        #     mask = mask.rotate(r)

        # 亮度、对比度、饱和度
        # p = random.random()
        # if p<0.2:
        #     # 亮度调整
        #     brightEnhancer = ImageEnhance.Brightness(image)
        #     image = brightEnhancer.enhance(random.uniform(0.8, 1.2))
        #     # 对比度调整
        #     contrastEnhancer = ImageEnhance.Contrast(image)
        #     image = contrastEnhancer.enhance(random.uniform(0.8, 1.2))
        #     # 饱和度调整
        #     colorEnhancer = ImageEnhance.Color(image)
        #     image = colorEnhancer.enhance(random.uniform(0.8, 2))

        # 裁剪
        p = random.random()
        if p < 0.2:
            w = random.randint(0, 99)
            h = random.randint(0, 99)
            image = image.crop((w, h, w + 316, h + 316))
            image = image.resize((416, 416), Image.ANTIALIAS)
            mask = mask.crop((w, h, w + 316, h + 316))
            mask = mask.resize((416, 416), Image.ANTIALIAS)

        # 以p进行透视变换
        p = random.random()
        if p < 0.2:
            # width, height = image.size
            # from_points = [(0, 0), (width, 0), (width, height), (0, height)]
            # new_points = [(width, 0 - 50),
            #               (0, 0),
            #               (0, height),
            #               (width, height + 50)]
            # params = find_coeffs(new_points, from_points)
            # image = image.transform((416, 416), Image.PERSPECTIVE, params, Image.BICUBIC)
            # mask = mask.transform((416, 416), Image.PERSPECTIVE, params)
            width, height = image.size
            from_points = [(0, 0), (width, 0), (width, height), (0, height)]
            new_points = [(width + random.randint(0, 50), 0 - random.randint(0, 50)),
                          (0 - random.randint(0, 50), 0 - random.randint(0, 50)),
                          (0 - random.randint(0, 50), height + random.randint(0, 50)),
                          (width + random.randint(0, 50), height + random.randint(0, 50))]
            params = find_coeffs(new_points, from_points)
            image = image.transform((416, 416), Image.PERSPECTIVE, params, Image.BICUBIC)
            mask = mask.transform((416, 416), Image.PERSPECTIVE, params)

        return image, mask

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        #print("size=",img.size,mask.size)
        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'


        if self.is_aug:
            img, mask = self.augment(img, mask)
        #img.show()
        #mask.show()

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            # 'image': torch.as_tensor(img.copy()).float().contiguous(),
            # 'mask': torch.as_tensor(mask.copy()).long().contiguous()
            'image': img.float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


