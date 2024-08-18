
import os
import numpy as np
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
import cv2
from glob import glob
import imageio
import torch.utils.data as data

class LiverDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.train_root = r"\dataset\liver\train"
        self.val_root = r"\liver\val"
        self.test_root = r"\liver\test"
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        assert self.state in ['train', 'val', 'test']
        if self.state == 'train':
            root = self.train_root
        elif self.state == 'val':
            root = self.val_root
        elif self.state == 'test':
            root = self.test_root

        pics = []
        masks = []
        n = len(os.listdir(root)) // 2  # 假设每个数据集包含图像和对应的mask
        for i in range(n):
            img = os.path.join(root, "%03d.png" % i)  # liverswinpre is %03d
            mask = os.path.join(root, "%03d_mask.png" % i)
            pics.append(img)
            masks.append(mask)
        return pics, masks

    def __getitem__(self, index):
        x_path = self.pics[index]
        y_path = self.masks[index]
        origin_x = Image.open(x_path)
        origin_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(origin_x)
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)
        return img_x, img_y
  # x_path, y_path
    def __len__(self):
        return len(self.pics)
class DriveEyeDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'D:\paperl\UNET-ZOO-master\dataset\DRIVE'
        self.pics, self.masks = self.getDataPath()
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.train_img_paths = glob(self.root + r'\training\images\*')
        self.train_mask_paths = glob(self.root + r'\training\1st_manual\*')
        self.val_img_paths = glob(self.root + r'\val\images\*')
        self.val_mask_paths = glob(self.root + r'\val\1st_manual\*')
        self.test_img_paths = glob(self.root + r'\test\images\*')
        self.test_mask_paths = glob(self.root + r'\test\1st_manual\\*')
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths, self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths, self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths, self.test_mask_paths

    def __getitem__(self, index):
        imgx,imgy=(512,512)
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        #print(pic_path)
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        if mask == None:
            mask = imageio.mimread(mask_path)
            mask = np.array(mask)[0]
        pic = cv2.resize(pic,(imgx,imgy))
        mask = cv2.resize(mask, (imgx, imgy))
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         deletemask = deletemask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         deletemask = deletemask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        # return img_x, img_y,pic_path,mask_path
        return img_x, img_y

    def __len__(self):
        return len(self.pics)

class ISIC2018_dataset(Dataset):
    def __init__(self, dataset_folder=r"D:\paperl\UNET-ZOO-master\data\ISIC2018_Task1_npy_all",
                 folder=r'D:\paperl\UNET-ZOO-master\Datasets\folder1', train_type='train', transform=None):
        self.transform = transform
        self.train_type = train_type
        self.folder_file = folder

        if self.train_type in ['train', 'validation', 'test']:
            list_name = 'folder1_' + self.train_type + '.list'
            # this is for cross validation
            list_path = os.path.join(self.folder_file, list_name)
            with open(list_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            self.folder = [join(dataset_folder, 'image', x) for x in self.image_list]
            self.mask = [join(dataset_folder, 'label', x.split('.')[0] + '_segmentation.npy') for x in self.image_list]
            # self.folder = sorted([join(dataset_folder, self.train_type, 'image', x) for x in
            #                       listdir(join(dataset_folder, self.train_type, 'image'))])
            # self.mask = sorted([join(dataset_folder, self.train_type, 'label', x) for x in
            #                     listdir(join(dataset_folder, self.train_type, 'label'))])
        else:
            print("Choosing type error, You have to choose the loading data type including: train, validation, test")

        assert len(self.folder) == len(self.mask)

    def __getitem__(self, item: int):
        image = np.load(self.folder[item])
        label = np.load(self.mask[item])

        sample = {'image': image, 'label': label}

        if self.transform is not None:
            # TODO: transformation to argument datasets
            sample = self.transform(sample, self.train_type)

        return sample['image'], sample['label']

    def __len__(self):
        return len(self.folder)

#Lung
class covid19(Dataset):
    def __init__(self, root, state='train', transform=None, mask_transform=None, img_ext='png', mask_ext='png'):
        self.root = root
        self.state = state
        self.transform = transform
        self.mask_transform = mask_transform
        self.img_ext = img_ext
        self.mask_ext = mask_ext

        # 获取数据路径并划分数据集
        self.img_paths, self.mask_paths = self.getDataPath()

    def getDataPath(self):
        split_file = os.path.join(self.root, 'split_indices.npz')
        if os.path.exists(split_file):
            splits = np.load(split_file, allow_pickle=True)
            train_img_paths = splits['train_img_paths']
            val_img_paths = splits['val_img_paths']
            test_img_paths = splits['test_img_paths']
            train_mask_paths = splits['train_mask_paths']
            val_mask_paths = splits['val_mask_paths']
            test_mask_paths = splits['test_mask_paths']
        else:
            img_pattern = os.path.join(self.root, 'images', f'*.{self.img_ext}')
            mask_pattern = os.path.join(self.root, 'masks', f'*.{self.mask_ext}')
            self.img_paths = glob(img_pattern)
            self.mask_paths = glob(mask_pattern)
            train_img_paths, test_img_paths, train_mask_paths, test_mask_paths = \
                train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
            train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
                train_test_split(train_img_paths, train_mask_paths, test_size=0.25, random_state=41)

            np.savez(split_file,
                     train_img_paths=train_img_paths,
                     val_img_paths=val_img_paths,
                     test_img_paths=test_img_paths,
                     train_mask_paths=train_mask_paths,
                     val_mask_paths=val_mask_paths,
                     test_mask_paths=test_mask_paths)

        if self.state == 'train':
            return train_img_paths, train_mask_paths
        elif self.state == 'val':
            return val_img_paths, val_mask_paths
        elif self.state == 'test':
            return test_img_paths, test_mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # image = image.convert("RGB")  # 转换为三通道图像


        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


