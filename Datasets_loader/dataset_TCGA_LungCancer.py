import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from PIL import Image
import os
import glob
from skimage import io
from tqdm import tqdm
import pandas as pd
from random import sample
from sklearn.utils import shuffle


class TCGA_LungCancer(torch.utils.data.Dataset):
    # @profile
    def __init__(self, train=True, transform=None, downsample=0.2, drop_threshold=0.1, preload=False, patch_downsample=1.0):
        self.train = train
        self.transform = transform
        self.downsample = downsample
        self.patch_downsample = patch_downsample
        self.drop_threshold = drop_threshold  # drop the pos slide of which positive patch ratio less than the threshold
        self.preload = preload
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        dir_LUAD = ""
        dir_LUSC = ""
        file_test_id = "/home/xiaoyuan/TCGA/TCGA_LungCancer/TEST_ID.csv"
        all_slides = glob.glob(dir_LUAD + "/*") + glob.glob(dir_LUSC + "/*")
        test_id = pd.read_csv(file_test_id)['0'].tolist()
        all_slides_train = []
        all_slides_test = []
        for slide_i in all_slides:
            if slide_i.split('/')[-1] not in test_id:
                all_slides_train.append(slide_i)
            else:
                all_slides_test.append(slide_i)
        all_slides = all_slides_train if train else all_slides_test
        # 1.1 down sample the slides
        print("================ Down sample ================")
        np.random.shuffle(all_slides)
        all_slides = all_slides[:int(len(all_slides)*self.downsample)]
        self.num_slides = len(all_slides)

        # 2.extract all available patches and build corresponding labels
        self.num_patches = 0
        if self.preload:
            self.all_patches = np.zeros([self.num_patches, 512, 512, 3], dtype=np.uint8)
        else:
            self.all_patches = []
        self.patch_corresponding_slide_label = []
        self.patch_corresponding_slide_index = []
        self.patch_corresponding_slide_name = []
        cnt_slide = 0
        cnt_patch = 0
        for i in tqdm(all_slides, ascii=True, desc='preload data'):
            all_patches_slide_i = glob.glob(os.path.join(i, '*.jpeg'))
            # all_patches_slide_i = glob.glob(os.path.join(i, '*/*.jpeg'))  # for 2x Mag
            if self.patch_downsample < 1.0:
                if int(len(all_patches_slide_i)*self.patch_downsample) > 3:
                    all_patches_slide_i = sample(all_patches_slide_i, int(len(all_patches_slide_i)*self.patch_downsample))
            for j in all_patches_slide_i:
                if self.preload:
                    self.all_patches[cnt_patch, :, :, :] = io.imread(j)
                else:
                    self.all_patches.append(j)
                self.patch_corresponding_slide_label.append(int('LUSC' in i.split('/')[-4]))
                self.patch_corresponding_slide_index.append(cnt_slide)
                self.patch_corresponding_slide_name.append(i.split('/')[-1])
                cnt_patch = cnt_patch + 1
            cnt_slide = cnt_slide + 1
        if not self.preload:
            self.all_patches = np.array(self.all_patches)
        self.patch_corresponding_slide_label = np.array(self.patch_corresponding_slide_label)
        self.patch_corresponding_slide_index = np.array(self.patch_corresponding_slide_index)
        self.patch_corresponding_slide_name = np.array(self.patch_corresponding_slide_name)

        self.num_patches = len(self.all_patches)
        # 3.do some statistics
        print("[DATA INFO] num_slide is {}; num_patches is {}\n".format(self.num_slides, self.num_patches))

    def __getitem__(self, index):
        if self.preload:
            patch_image = self.all_patches[index]
        else:
            patch_image = io.imread(self.all_patches[index])
        patch_corresponding_slide_label = self.patch_corresponding_slide_label[index]
        patch_corresponding_slide_index = self.patch_corresponding_slide_index[index]
        patch_corresponding_slide_name = self.patch_corresponding_slide_name[index]

        patch_image = self.transform(Image.fromarray(np.uint8(patch_image), 'RGB'))
        patch_label = 0  # patch_label is not available in TCGA
        return patch_image, [patch_label, patch_corresponding_slide_label, patch_corresponding_slide_index,
                             patch_corresponding_slide_name], index

    def __len__(self):
        return self.num_patches


class TCGA_LungCancer_Feat(torch.utils.data.Dataset):
    # @profile
    def __init__(self, train=True, downsample=1.0, return_bag=False):
        self.train = train
        self.return_bag = return_bag
        bags_csv = ''
        bags_path = pd.read_csv(bags_csv)
        train_path = bags_path.iloc[0:int(len(bags_path) * 0.8), :]
        test_path = bags_path.iloc[int(len(bags_path) * 0.8):, :]
        train_path = shuffle(train_path).reset_index(drop=True)
        test_path = shuffle(test_path).reset_index(drop=True)

        if downsample < 1.0:
            train_path = train_path.iloc[0:int(len(train_path) * downsample), :]
            test_path = test_path.iloc[0:int(len(test_path) * downsample), :]

        self.patch_feat_all = []
        self.patch_corresponding_slide_label = []
        self.patch_corresponding_slide_index = []
        self.patch_corresponding_slide_name = []
        if self.train:
            for i in tqdm(range(len(train_path)), desc='loading data'):
                label, feats = get_bag_feats(train_path.iloc[i])
                self.patch_feat_all.append(feats)
                self.patch_corresponding_slide_label.append(np.ones(feats.shape[0]) * label)
                self.patch_corresponding_slide_index.append(np.ones(feats.shape[0]) * i)
                self.patch_corresponding_slide_name.append(np.ones(feats.shape[0]) * i)
        else:
            for i in tqdm(range(len(test_path)), desc='loading data'):
                label, feats = get_bag_feats(test_path.iloc[i])
                self.patch_feat_all.append(feats)
                self.patch_corresponding_slide_label.append(np.ones(feats.shape[0]) * label)
                self.patch_corresponding_slide_index.append(np.ones(feats.shape[0]) * i)
                self.patch_corresponding_slide_name.append(np.ones(feats.shape[0]) * i)

        self.patch_feat_all = np.concatenate(self.patch_feat_all, axis=0).astype(np.float32)
        self.patch_corresponding_slide_label = np.concatenate(self.patch_corresponding_slide_label).astype(np.long)
        self.patch_corresponding_slide_index =np.concatenate(self.patch_corresponding_slide_index).astype(np.long)
        self.patch_corresponding_slide_name = np.concatenate(self.patch_corresponding_slide_name)

        self.num_patches = self.patch_feat_all.shape[0]
        self.patch_label_all = np.zeros([self.patch_feat_all.shape[0]], dtype=np.long)  # Patch label is not available and set to 0 !
        # 3.do some statistics
        print("[DATA INFO] num_slide is {}; num_patches is {}\n".format(len(train_path), self.num_patches))

    def __getitem__(self, index):
        if self.return_bag:
            idx_patch_from_slide_i = np.where(self.patch_corresponding_slide_index == index)[0]
            bag = self.patch_feat_all[idx_patch_from_slide_i, :]
            patch_labels = self.patch_label_all[idx_patch_from_slide_i]  # Patch label is not available and set to 0 !
            slide_label = self.patch_corresponding_slide_label[idx_patch_from_slide_i][0]
            slide_index = self.patch_corresponding_slide_index[idx_patch_from_slide_i][0]
            slide_name = self.patch_corresponding_slide_name[idx_patch_from_slide_i][0]

            # check data
            if self.patch_corresponding_slide_label[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_label[idx_patch_from_slide_i].min():
                raise
            if self.patch_corresponding_slide_index[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_index[idx_patch_from_slide_i].min():
                raise
            return bag, [patch_labels, slide_label, slide_index, slide_name], index
        else:
            patch_image = self.patch_feat_all[index]
            patch_corresponding_slide_label = self.patch_corresponding_slide_label[index]
            patch_corresponding_slide_index = self.patch_corresponding_slide_index[index]
            patch_corresponding_slide_name = self.patch_corresponding_slide_name[index]

            patch_label = self.patch_label_all[index]  # Patch label is not available and set to 0 !
            return patch_image, [patch_label, patch_corresponding_slide_label, patch_corresponding_slide_index,
                                 patch_corresponding_slide_name], index

    def __len__(self):
        if self.return_bag:
            return self.patch_corresponding_slide_index.max() + 1
        else:
            return self.num_patches


def get_bag_feats(csv_file_df):
    feats_csv_path = '' + csv_file_df.iloc[0].split('/')[1] + '.csv'
    df = pd.read_csv(feats_csv_path)
    feats = shuffle(df).reset_index(drop=True)
    feats = feats.to_numpy()
    label = np.zeros(1)
    label[0] = csv_file_df.iloc[1]
    return label, feats


if __name__ == '__main__':
    train_ds_feat = TCGA_LungCancer_Feat(train=True, downsample=1.0)
    test_ds_feat = TCGA_LungCancer_Feat(train=False, downsample=1.0)

    trans = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor()])
    train_ds = TCGA_LungCancer(train=True, transform=None, downsample=0.1, drop_threshold=0, preload=False)
    val_ds = TCGA_LungCancer(train=False, transform=None, downsample=0.1, drop_threshold=0, preload=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64,
                                               shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64,
                                             shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
    for data in tqdm(train_loader, desc='loading'):
        patch_img = data[0]
        label_patch = data[1][0]
        label_bag = data[1][1]
        idx = data[-1]
    print("END")
