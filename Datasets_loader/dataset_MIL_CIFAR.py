"""Pytorch Dataset object that loads perfectly balanced MNIST dataset in bag form."""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from PIL import Image

import numpy as np
from six.moves import cPickle as pickle
import os
import platform
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3072).reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs, axis=0)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data():
    # Load the raw CIFAR-10 data
    cifar10_dir = ''
    x_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    x_train, y_train, X_test, y_test = torch.from_numpy(x_train), torch.from_numpy(y_train), \
                                       torch.from_numpy(X_test), torch.from_numpy(y_test)
    return x_train, y_train, X_test, y_test


def random_shuffle(input_tensor):
    length = input_tensor.shape[0]
    random_idx = torch.randperm(length)
    output_tensor = input_tensor[random_idx]
    return output_tensor


class CIFAR_WholeSlide_challenge(torch.utils.data.Dataset):
    def __init__(self, train, positive_num=[8, 9], negative_num=[0, 1, 2, 3, 4, 5, 6, 7],
                 bag_length=10, return_bag=False, num_img_per_slide=600, pos_patch_ratio=0.1, pos_slide_ratio=0.5, transform=None, accompanyPos=True):
        self.train = train
        self.positive_num = positive_num  # transform the N-class into 2-class
        self.negative_num = negative_num  # transform the N-class into 2-class
        self.bag_length = bag_length
        self.return_bag = return_bag  # return patch ot bag
        self.transform = transform    # transform the patch image
        self.num_img_per_slide = num_img_per_slide

        if train:
            self.ds_data, self.ds_label, _, _ = get_CIFAR10_data()
            try:
                self.ds_data_simCLR_feat = torch.from_numpy(np.load("./Datasets_loader/all_feats_CIFAR.npy")[:50000, :]).float()
                print("Pre-trained feat found")
            except:
                print("No pre-trained feat found")
        else:
            _, _ , self.ds_data, self.ds_label = get_CIFAR10_data()
            try:
                self.ds_data_simCLR_feat = torch.from_numpy(np.load("./Datasets_loader/all_feats_CIFAR.npy")[50000:, :]).float()
                print("Pre-trained feat found")
            except:
                print("No pre-trained feat found")

        self.build_whole_slides(num_img=num_img_per_slide, positive_nums=positive_num, negative_nums=negative_num, pos_patch_ratio=pos_patch_ratio, pos_slide_ratio=pos_slide_ratio)
        print("")

    def build_whole_slides(self, num_img, positive_nums, negative_nums, pos_patch_ratio=0.1, pos_slide_ratio=0.5):
        # num_img: num of images per slide
        # positive patch ratio in each slide

        num_pos_per_slide = int(num_img * pos_patch_ratio)
        num_neg_per_slide = num_img - num_pos_per_slide

        idx_pos = []
        for num in positive_nums:
            idx_pos.append(torch.where(self.ds_label == num)[0])
        idx_pos = torch.cat(idx_pos).unsqueeze(1)
        idx_neg = []
        for num in negative_nums:
            idx_neg.append(torch.where(self.ds_label == num)[0])
        idx_neg = torch.cat(idx_neg).unsqueeze(1)

        idx_pos = random_shuffle(idx_pos)
        idx_neg = random_shuffle(idx_neg)

        # build pos slides using calculated
        num_pos_2PosSlides = int(idx_neg.numel() // ((1 - pos_slide_ratio) / (pos_patch_ratio*pos_slide_ratio) + (1 - pos_patch_ratio) / pos_patch_ratio))
        if num_pos_2PosSlides > idx_pos.shape[0]:
            num_pos_2PosSlides = idx_pos.shape[0]
        num_pos_2PosSlides = int(num_pos_2PosSlides // num_pos_per_slide * num_pos_per_slide)
        num_neg_2PosSlides = int(num_pos_2PosSlides * ((1-pos_patch_ratio)/pos_patch_ratio))
        num_neg_2NegSlides = int(num_pos_2PosSlides * ((1-pos_slide_ratio)/(pos_patch_ratio*pos_slide_ratio)))

        num_neg_2PosSlides = int(num_neg_2PosSlides // num_neg_per_slide * num_neg_per_slide)
        num_neg_2NegSlides = int(num_neg_2NegSlides // num_img * num_img)

        if num_neg_2PosSlides // num_neg_per_slide != num_pos_2PosSlides // num_pos_per_slide :
            num_diff_slide = num_pos_2PosSlides // num_pos_per_slide - num_neg_2PosSlides // num_neg_per_slide
            num_pos_2PosSlides = num_pos_2PosSlides - num_pos_per_slide * num_diff_slide

        idx_pos = idx_pos[0:num_pos_2PosSlides]
        idx_neg = idx_neg[0:(num_neg_2PosSlides+num_neg_2NegSlides)]

        idx_pos_toPosSlide = idx_pos[:].reshape(-1, num_pos_per_slide)
        idx_neg_toPosSlide = idx_neg[0:num_neg_2PosSlides].reshape(-1, num_neg_per_slide)
        idx_neg_toNegSlide = idx_neg[num_neg_2PosSlides:].reshape(-1, num_img)

        idx_pos_slides = torch.cat([idx_pos_toPosSlide, idx_neg_toPosSlide], dim=1)
        # idx_pos_slides = idx_pos_slides[:, torch.randperm(idx_pos_slides.shape[1])]  #  shuffle pos and neg idx
        for i_ in range(idx_pos_slides.shape[0]):
            idx_pos_slides[i_, :] = idx_pos_slides[i_, torch.randperm(idx_pos_slides.shape[1])]
        idx_neg_slides = idx_neg_toNegSlide

        self.idx_all_slides = torch.cat([idx_pos_slides, idx_neg_slides], dim=0)
        self.label_all_slides = torch.cat([torch.ones(idx_pos_slides.shape[0]), torch.zeros(idx_neg_slides.shape[0])], dim=0)
        self.label_all_slides = self.label_all_slides.unsqueeze(1).repeat([1,self.idx_all_slides.shape[1]]).long()
        print("[Info] dataset: {}".format(self.idx_all_slides.shape))
        #self.visualize(idx_pos_slides[0])

    def __getitem__(self, index):
        if self.return_bag:
            bagPerSlide = self.idx_all_slides.shape[1] // self.bag_length
            idx_slide = index // bagPerSlide
            idx_bag_in_slide = index % bagPerSlide
            idx_images = self.idx_all_slides[idx_slide, (idx_bag_in_slide*self.bag_length):((idx_bag_in_slide+1)*self.bag_length)]
            bag = self.ds_data[idx_images]
            patch_labels_raw = self.ds_label[idx_images]
            patch_labels = torch.zeros_like(patch_labels_raw)
            for num in self.positive_num:
                patch_labels[patch_labels_raw == num] = 1
            patch_labels = patch_labels.long()
            slide_label = self.label_all_slides[idx_slide, 0]
            slide_name = str(idx_slide)
            return bag.float()/255, [patch_labels, slide_label, idx_slide, slide_name], index
        else:
            idx_image = self.idx_all_slides.flatten()[index]
            slide_label = self.label_all_slides.flatten()[index]
            idx_slide = index // self.num_img_per_slide
            slide_name = str(idx_slide)
            patch = self.ds_data[idx_image]
            patch_label = self.ds_label[idx_image]
            patch_label = int(patch_label in self.positive_num)
            return patch.float()/255, [patch_label, slide_label, idx_slide, slide_name], index

    def __len__(self):
        if self.return_bag:
            return self.idx_all_slides.shape[1] // self.bag_length * self.idx_all_slides.shape[0]
        else:
            return self.idx_all_slides.numel()

    def visualize(self, idx, number_row=20, number_col=30):
        # idx should be of shape num_img_per_slide
        slide = self.ds_data[idx].clone()  # num_img_per_slide * 3 * 32 * 32
        patch_label = self.ds_label[idx].clone()
        idx_pos_patch = []
        for num in self.positive_num:
            idx_pos_patch.append(torch.where(patch_label == num)[0])
        idx_pos_patch = torch.cat(idx_pos_patch)
        slide[idx_pos_patch, 0, :2, :] = 255
        slide[idx_pos_patch, 0, -2:, :] = 255
        slide[idx_pos_patch, 0, :, :2] = 255
        slide[idx_pos_patch, 0, :, -2:] = 255

        slide[idx_pos_patch, 1, :2, :] = 0
        slide[idx_pos_patch, 1, -2:, :] = 0
        slide[idx_pos_patch, 1, :, :2] = 0
        slide[idx_pos_patch, 1, :, -2:] = 0

        slide[idx_pos_patch, 2, :2, :] = 0
        slide[idx_pos_patch, 2, -2:, :] = 0
        slide[idx_pos_patch, 2, :, :2] = 0
        slide[idx_pos_patch, 2, :, -2:] = 0

        slide = slide.unsqueeze(0).reshape(number_row, number_col, 3, 32, 32).permute(0, 3, 1, 4, 2).reshape(number_row*32, number_col*32, 3)
        import utliz
        utliz.show_img(slide)
        return 0


if __name__ == "__main__":
    # Invoke the above function to get our data.
    x_train, y_train,x_test, y_test = get_CIFAR10_data()

    # for pos_slide_ratio in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
    for pos_slide_ratio in [0.01, 0.05, 0.1, 0.2, 0.5, 0.7]:
        print("=========== pos slide ratio: {} ===========".format(pos_slide_ratio))
        train_ds = CIFAR_WholeSlide_challenge(train=True, positive_num=[9], negative_num=[0, 1, 2, 3, 4, 5, 6, 7, 8], bag_length=100, return_bag=False, num_img_per_slide=100, pos_patch_ratio=pos_slide_ratio, pos_slide_ratio=0.5, transform=None)
        train_loader = data_utils.DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
        test_ds_part1 = CIFAR_WholeSlide_challenge(train=False, positive_num=[9], negative_num=[0, 1, 2, 3, 4, 5, 6, 7, 8], bag_length=100, return_bag=False, num_img_per_slide=100, pos_patch_ratio=pos_slide_ratio, pos_slide_ratio=0.5, transform=None)
        test_loader_part1 = data_utils.DataLoader(test_ds_part1, batch_size=64, shuffle=True, drop_last=False)
        print("")
    print("")

