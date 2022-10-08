"""Pytorch Dataset object that loads perfectly balanced MNIST dataset in bag form."""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from six.moves import cPickle as pickle
import os
from sklearn.model_selection import train_test_split
import platform


Patho_classes = ['NORM', 'MUC', 'TUM', 'STR', 'LYM', 'BACK', 'MUS', 'DEB', 'ADI']


def load_Pathdata(Root, downsample_ratio=1.0):
    X = []
    Y = []
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for index_class, folder in enumerate(Patho_classes):
        all_files_class_i = os.listdir(os.path.join(Root, folder))
        all_files_class_i = all_files_class_i[:int(len(all_files_class_i)*downsample_ratio)]
        for file in tqdm(all_files_class_i):
            image = np.array(Image.open(os.path.join(Root, folder, file)).convert("RGB")).transpose(2, 0, 1)
            # image = np.array(cv2.imread(os.path.join(Root, folder, file)),cv2.IMREAD_UNCHANGED)
            label = index_class
            X.append(image)
            Y.append(label)
        train_X, test_X, train_Y, test_Y = train_test_split(np.array(X), np.array(Y), test_size=0.2, random_state=42)
        X_train.append(train_X)
        X_test.append(test_X)
        Y_train.append(train_Y)
        Y_test.append(test_Y)
        X=[]
        Y=[]
    X_train = [b for a in X_train for b in a]
    X_test = [b for a in X_test for b in a]
    Y_train = [b for a in Y_train for b in a]
    Y_test = [b for a in Y_test for b in a]

    return np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)


def get_Path10_data(Path10_dir='', downsample_ratio=1.0):
    # Load the raw Path-10 data
    X_train, X_test, Y_train, Y_test = load_Pathdata(Path10_dir, downsample_ratio)
    x_train, y_train, X_test, y_test = torch.from_numpy(X_train), torch.from_numpy(Y_train), \
                                       torch.from_numpy(X_test), torch.from_numpy(Y_test)
    return x_train, y_train, X_test, y_test


def random_shuffle(input_tensor):
    length = input_tensor.shape[0]
    random_idx = torch.randperm(length)
    output_tensor = input_tensor[random_idx]
    return output_tensor


class NCT_WholeSlide_challenge(torch.utils.data.Dataset):
    def __init__(self, ds_data, ds_label, positive_num=[2], negative_num=[0, 1, 3, 4, 5, 6, 7, 8],
                 bag_length=10, return_bag=False, num_img_per_slide=600, pos_patch_ratio=0.1, pos_slide_ratio=0.5, transform=None):

        self.positive_num = positive_num  # transform the N-class into 2-class
        self.negative_num = negative_num  # transform the N-class into 2-class
        self.bag_length = bag_length
        self.return_bag = return_bag  # return patch ot bag
        self.transform = transform    # transform the patch image
        self.num_img_per_slide = num_img_per_slide

        self.ds_data, self.ds_label = ds_data, ds_label
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

    def visualize(self, idx, number_row=10, number_col=10):
        # idx should be of shape num_img_per_slide
        slide = self.ds_data[idx].clone()  # num_img_per_slide * 3 * 32 * 32
        patch_label = self.ds_label[idx].clone()
        idx_pos_patch = []
        for num in self.positive_num:
            idx_pos_patch.append(torch.where(patch_label == num)[0])
        idx_pos_patch = torch.cat(idx_pos_patch)
        slide[idx_pos_patch, 0, :10, :] = 255
        slide[idx_pos_patch, 0, -10:, :] = 255
        slide[idx_pos_patch, 0, :, :10] = 255
        slide[idx_pos_patch, 0, :, -10:] = 255

        slide[idx_pos_patch, 1, :10, :] = 0
        slide[idx_pos_patch, 1, -10:, :] = 0
        slide[idx_pos_patch, 1, :, :10] = 0
        slide[idx_pos_patch, 1, :, -10:] = 0

        slide[idx_pos_patch, 2, :10, :] = 0
        slide[idx_pos_patch, 2, -10:, :] = 0
        slide[idx_pos_patch, 2, :, :10] = 0
        slide[idx_pos_patch, 2, :, -10:] = 0

        slide = slide.unsqueeze(0).reshape(number_row, number_col, 3, 224, 224).permute(0, 3, 1, 4, 2).reshape(number_row*224, number_col*224, 3)
        import utliz
        # show_img_1(slide)
        return slide


def show_img_1(img, save_file_name=''):
    if type(img) == torch.Tensor:
        img = img.cpu().detach().numpy()
    if len(img.shape) == 3:  # HxWx3 or 3xHxW, treat as RGB image
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
    fig = plt.figure()
    plt.imshow(img)
    if save_file_name != '':
        plt.savefig(save_file_name, format='svg')
    plt.colorbar()
    plt.show()


def show_img(img, save_file_name='',format='svg', dpi=1200):
    if type(img) == torch.Tensor:
        img = img.cpu().detach().numpy()
    if len(img.shape) == 3:  # HxWx3 or 3xHxW, treat as RGB image
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
    fig = plt.figure()
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    if save_file_name != '':
        plt.savefig(save_file_name, format=format, dpi=dpi, pad_inches=0.0, bbox_inches='tight')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    # train_data, train_label, val_data, val_label = get_Path10_data(downsample_ratio=0.1)
    # for pos_slide_ratio in [0.01, 0.05, 0.1, 0.2, 0.5, 0.7]:
    #     print("=========== pos slide ratio: {} ===========".format(pos_slide_ratio))
    #     train_ds = NCT_WholeSlide_challenge(ds_data=train_data, ds_label=train_label, positive_num=[2], negative_num=[0, 1, 3, 4, 5, 6, 7, 8], bag_length=100, return_bag=False, num_img_per_slide=100, pos_patch_ratio=pos_slide_ratio, pos_slide_ratio=0.5, transform=None)
    #     slide = train_ds.visualize(train_ds.idx_all_slides[0])
    #     show_img(slide, save_file_name='../figures/NCT_WSI_pos_PPR{}.png'.format(pos_slide_ratio), format='png', dpi=2400)
    #     slide = train_ds.visualize(train_ds.idx_all_slides[-10])
    #     show_img(slide, save_file_name='../figures/NCT_WSI_neg_PPR{}.png'.format(pos_slide_ratio), format='png', dpi=2400)
    #     # print("")
    # print("")

    train_data, train_label, val_data, val_label = get_Path10_data(downsample_ratio=1.0)
    for pos_slide_ratio in [0.01, 0.05, 0.1, 0.2, 0.5, 0.7]:
        print("=========== pos slide ratio: {} ===========".format(pos_slide_ratio))
        train_ds = NCT_WholeSlide_challenge(ds_data=train_data, ds_label=train_label, positive_num=[2], negative_num=[0, 1, 3, 4, 5, 6, 7, 8], bag_length=100, return_bag=False, num_img_per_slide=100, pos_patch_ratio=pos_slide_ratio, pos_slide_ratio=0.5, transform=None)
        val_ds = NCT_WholeSlide_challenge(ds_data=val_data, ds_label=val_label, positive_num=[2], negative_num=[0, 1, 3, 4, 5, 6, 7, 8], bag_length=100, return_bag=False, num_img_per_slide=100, pos_patch_ratio=pos_slide_ratio, pos_slide_ratio=0.5, transform=None)
        print("")
    print("")

