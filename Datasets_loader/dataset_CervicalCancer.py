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


def shuffle_downsample_myself(input_list, downsample):
    np.random.shuffle(input_list)
    if downsample == 1:
        output_list = input_list
    elif downsample < 1:
        output_list = input_list[:int(len(input_list) * downsample)]
    elif downsample > 1:
        downsample = int(downsample)
        if downsample > len(input_list):
            output_list = input_list
        else:
            output_list = input_list[:downsample]
    else:
        raise
    return output_list


class CervicalCaner_16(torch.utils.data.Dataset):
    # @profile
    def __init__(self, root_dir="",
                 train=True, transform=None, downsample=1.0, preload=True, return_bag=False):
        self.root_dir = root_dir
        self.train = train
        self.return_bag = return_bag
        self.transform = transform
        self.downsample = downsample
        self.preload = preload
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        if train:
            self.root_dir = os.path.join(self.root_dir, "train_datasets")
        else:
            self.root_dir = os.path.join(self.root_dir, "test_datasets")

        all_slides = glob.glob(self.root_dir + "/*/*")

        self.num_slides = len(all_slides)

        # 2.extract all available patches and build corresponding labels
        if self.preload:
            self.num_patches = 0
            for i in tqdm(all_slides, ascii=True, desc='scanning data'):
                self.num_patches = self.num_patches + len(shuffle_downsample_myself(os.listdir(i), self.downsample))
            self.all_patches = np.zeros([self.num_patches, 224, 224, 3], dtype=np.uint8)
        else:
            self.all_patches = []
        self.patch_label = []
        self.patch_corresponding_slide_label = []
        self.patch_corresponding_slide_index = []
        self.patch_corresponding_slide_name = []
        cnt_slide = 0
        cnt_patch = 0
        for i in tqdm(all_slides, ascii=True, desc='preload data'):
            patches_from_slide_i = os.listdir(i)
            patches_from_slide_i = shuffle_downsample_myself(patches_from_slide_i, self.downsample)
            for j in patches_from_slide_i:
                if self.preload:
                    self.all_patches[cnt_patch, :, :, :] = io.imread(os.path.join(i, j))
                else:
                    self.all_patches.append(os.path.join(i, j))
                self.patch_label.append(0)
                self.patch_corresponding_slide_label.append(int('P' == i.split('/')[-2]))
                self.patch_corresponding_slide_index.append(cnt_slide)
                self.patch_corresponding_slide_name.append(i.split('/')[-1])
                cnt_patch = cnt_patch + 1
            cnt_slide = cnt_slide + 1
        if not self.preload:
            self.all_patches = np.array(self.all_patches)
            self.num_patches = self.all_patches.shape[0]
        self.all_patches = self.all_patches.transpose(0, 3, 1, 2)
        self.patch_label = np.array(self.patch_label, dtype=np.long)  # [Attention] patch label is unavailable and set to 0
        self.patch_corresponding_slide_label = np.array(self.patch_corresponding_slide_label, dtype=np.long)
        self.patch_corresponding_slide_index = np.array(self.patch_corresponding_slide_index, dtype=np.long)
        self.patch_corresponding_slide_name = np.array(self.patch_corresponding_slide_name)

        # 3.do some statistics
        print("[DATA INFO] num_slide is {}; num_patches is {}\npos_patch_ratio is unknown".format(
            self.num_slides, self.num_patches))
        print("")

    def __getitem__(self, index):
        if self.return_bag:
            idx_patch_from_slide_i = np.where(self.patch_corresponding_slide_index == index)[0]
            bag = self.all_patches[idx_patch_from_slide_i, :, :, :]
            bag = bag.astype(np.float32)/255
            patch_labels = self.patch_label[idx_patch_from_slide_i]  # Patch label is not available and set to 0 !
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
            if self.preload:
                patch_image = self.all_patches[index]
            else:
                patch_image = io.imread(self.all_patches[index])
            patch_label = self.patch_label[index]  # [Attention] patch label is unavailable and set to 0
            patch_corresponding_slide_label = self.patch_corresponding_slide_label[index]
            patch_corresponding_slide_index = self.patch_corresponding_slide_index[index]
            patch_corresponding_slide_name = self.patch_corresponding_slide_name[index]

            # patch_image = self.transform(Image.fromarray(np.uint8(patch_image), 'RGB'))
            patch_image = patch_image.astype(np.float32)/255
            return patch_image, [patch_label, patch_corresponding_slide_label, patch_corresponding_slide_index,
                                 patch_corresponding_slide_name], index

    def __len__(self):
        if self.return_bag:
            return self.patch_corresponding_slide_index.max() + 1
        else:
            return self.num_patches


class CervicalCaner_16_feat(torch.utils.data.Dataset):
    def __init__(self, root_dir="",
                 train=True, return_bag=True):
        self.root_dir = root_dir
        self.train = train
        self.return_bag = return_bag

        # 1. load all featreus and slide label and index
        save_path = ""
        if train:
            self.all_patches = np.load(os.path.join(save_path, "train_feats.npy"))
            self.patch_corresponding_slide_label = np.load(os.path.join(save_path, "train_corresponding_slide_label.npy"))
            self.patch_corresponding_slide_index = np.load(os.path.join(save_path, "train_corresponding_slide_index.npy"))
            self.patch_label = np.zeros_like(self.patch_corresponding_slide_label)  # [Attention] patch label is unavailable and set to 0
            self.patch_corresponding_slide_name = self.patch_corresponding_slide_index
        else:
            self.all_patches = np.load(os.path.join(save_path, "test_feats.npy"))
            self.patch_corresponding_slide_label = np.load(os.path.join(save_path, "test_corresponding_slide_label.npy"))
            self.patch_corresponding_slide_index = np.load(os.path.join(save_path, "test_corresponding_slide_index.npy"))
            self.patch_label = np.zeros_like(self.patch_corresponding_slide_label)  # [Attention] patch label is unavailable and set to 0
            self.patch_corresponding_slide_name = self.patch_corresponding_slide_index

        self.num_patches = self.all_patches.shape[0]
        self.num_slides = self.patch_corresponding_slide_index.max() + 1
        print("[DATA INFO] num_slide is {}; num_patches is {}\npos_patch_ratio is unknown".format(
            self.num_slides, self.num_patches))

        # 2. sort instances features into bag
        self.slide_feat_all = []
        self.slide_label_all = []
        self.slide_patch_label_all = []
        for i in range(self.num_slides):
            idx_from_same_slide = self.patch_corresponding_slide_index == i
            idx_from_same_slide = np.nonzero(idx_from_same_slide)[0]

            self.slide_feat_all.append(self.all_patches[idx_from_same_slide])
            if self.patch_corresponding_slide_label[idx_from_same_slide].max() != self.patch_corresponding_slide_label[idx_from_same_slide].min():
                raise
            self.slide_label_all.append(self.patch_corresponding_slide_label[idx_from_same_slide].max())
            self.slide_patch_label_all.append(np.zeros(idx_from_same_slide.shape[0]).astype(np.long))
        print("")

    def __getitem__(self, index):
        if self.return_bag:
            slide_feat = self.slide_feat_all[index]
            slide_label = self.slide_label_all[index]
            slide_patch_label = self.slide_patch_label_all[index]
            return slide_feat, [slide_patch_label, slide_label], index
        else:
            patch_image_feat = self.all_patches[index]
            patch_label = self.patch_label[index]  # [Attention] patch label is unavailable and set to 0
            patch_corresponding_slide_label = self.patch_corresponding_slide_label[index]
            patch_corresponding_slide_index = self.patch_corresponding_slide_index[index]
            patch_corresponding_slide_name = self.patch_corresponding_slide_name[index]

            return patch_image_feat, [patch_label, patch_corresponding_slide_label, patch_corresponding_slide_index,
                                      patch_corresponding_slide_name], index

    def __len__(self):
        if self.return_bag:
            return self.num_slides
        else:
            return self.num_patches


if __name__ == '__main__':
    train_ds_return_bag = CervicalCaner_16_feat(train=True, return_bag=True)
    train_ds_return_instance = CervicalCaner_16_feat(train=True, return_bag=False)

    train_ds = CervicalCaner_16(root_dir=root_dir, train=True, transform=None, downsample=10, preload=True)
    val_ds = CervicalCaner_16(root_dir=root_dir, train=False, transform=None, downsample=10, preload=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64,
                                               shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64,
                                             shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
    for data in train_loader:
        patch_img = data[0]
        label_patch = data[1][0]
        label_bag = data[1][1]
        idx = data[-1]
    print("END")
