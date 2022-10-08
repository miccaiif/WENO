import argparse
import warnings
import os
import time
import numpy as np

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
from tensorboardX import SummaryWriter
# import models
# from models.alexnet import alexnet_CIFAR10, alexnet_CIFAR10_Attention
from models.resnetv1 import resnet_NCT_Encoder, teacher_Attention_head, student_head
# from dataset_toy import Dataset_toy
# from Datasets_loader.dataset_MNIST_challenge import MNIST_WholeSlide_challenge
# from Datasets_loader.dataset_MIL_NCTCRCHE import NCT_WholeSlide_challenge, get_Path10_data
from Datasets_loader.dataset_CervicalCancer import CervicalCaner_16
import datetime
import utliz
import util
import random
from tqdm import tqdm
import copy


class Optimizer:
    def __init__(self, model_encoder, model_teacherHead, model_studentHead,
                 optimizer_encoder, optimizer_teacherHead, optimizer_studentHead,
                 train_bagloader, train_instanceloader, test_bagloader, test_instanceloader,
                 writer=None, num_epoch=100,
                 dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 PLPostProcessMethod='NegGuide', StuFilterType='ReplaceAS', smoothE=100,
                 stu_loss_weight_neg=0.1, stuOptPeriod=1):
        self.model_encoder = model_encoder
        self.model_teacherHead = model_teacherHead
        self.model_studentHead = model_studentHead
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_teacherHead = optimizer_teacherHead
        self.optimizer_studentHead = optimizer_studentHead
        self.train_bagloader = train_bagloader
        self.train_instanceloader = train_instanceloader
        self.test_bagloader = test_bagloader
        self.test_instanceloader = test_instanceloader
        self.writer = writer
        self.num_epoch = num_epoch
        self.dev = dev
        self.log_period = 10
        self.PLPostProcessMethod = PLPostProcessMethod
        self.StuFilterType = StuFilterType
        self.smoothE = smoothE
        self.stu_loss_weight_neg = stu_loss_weight_neg
        self.stuOptPeriod = stuOptPeriod

    def optimize(self):
        self.Bank_all_Bags_label = None
        self.Bank_all_instances_pred_byTeacher = None
        self.Bank_all_instances_feat_byTeacher = None
        self.Bank_all_instances_pred_processed = None

        self.Bank_all_instances_pred_byStudent = None

        # Load pre-extracted SimCLR features
        # pre_trained_SimCLR_feat = self.train_instanceloader.dataset.ds_data_simCLR_feat[self.train_instanceloader.dataset.idx_all_slides].to(self.dev)
        for epoch in range(self.num_epoch):
            self.optimize_teacher(epoch)
            self.evaluate_teacher(epoch)
            if epoch % self.stuOptPeriod == 0:
                self.optimize_student(epoch)
                self.evaluate_student(epoch)
        return 0

    def optimize_teacher(self, epoch):
        self.model_encoder.train()
        self.model_teacherHead.train()
        self.model_studentHead.eval()
        ## optimize teacher with bag-dataloader
        # 1. change loader to bag-loader
        loader = self.train_bagloader
        # 2. optimize
        patch_label_gt = []
        patch_label_pred = []
        bag_label_gt = []
        bag_label_pred = []
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Teacher training')):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)
            feat = self.model_encoder(data.squeeze(0))
            if epoch > self.smoothE:
                if "FilterNegInstance" in self.StuFilterType:
                    # using student prediction to remove negative instance feat in the positive bag
                    if label[1] == 1:
                        with torch.no_grad():
                            pred_byStudent = self.model_studentHead(feat)
                            pred_byStudent = torch.softmax(pred_byStudent, dim=1)[:, 1]
                        if '_Top' in self.StuFilterType:
                            # strategy A: remove the topK most negative instance
                            idx_to_keep = torch.topk(-pred_byStudent, k=int(self.StuFilterType.split('_Top')[-1]))[1]
                        elif '_ThreProb' in self.StuFilterType:
                            # strategy B: remove the negative instance above prob K
                            idx_to_keep = torch.where(pred_byStudent >= int(self.StuFilterType.split('_Thre')[-1])/100.0)[0]
                        feat_removedNeg = feat[idx_to_keep]
                        bag_prediction, _, _, instance_attn_score = self.model_teacherHead(feat_removedNeg, returnBeforeSoftMaxA=True, scores_replaceAS=None)
                        instance_attn_score = torch.cat([instance_attn_score, instance_attn_score.min()*torch.ones(1, feat.shape[0]-instance_attn_score.shape[0]).to(instance_attn_score.device)], dim=0)
                        # with torch.no_grad():
                        #     _, _, _, instance_attn_score = self.model_teacherHead(feat, returnBeforeSoftMaxA=True, scores_replaceAS=None)
                    else:
                        bag_prediction, _, _, instance_attn_score = self.model_teacherHead(feat, returnBeforeSoftMaxA=True, scores_replaceAS=None)
                else:
                    bag_prediction, _, _, instance_attn_score = self.model_teacherHead(feat, returnBeforeSoftMaxA=True, scores_replaceAS=None)
            else:
                bag_prediction, _, _, instance_attn_score = self.model_teacherHead(feat, returnBeforeSoftMaxA=True, scores_replaceAS=None)
            bag_prediction = torch.softmax(bag_prediction, 1)
            loss_teacher = -1. * (label[1] * torch.log(bag_prediction[0, 1]+1e-5) + (1. - label[1]) * torch.log(1. - bag_prediction[0, 1]+1e-5))
            self.optimizer_encoder.zero_grad()
            self.optimizer_teacherHead.zero_grad()
            loss_teacher.backward()
            self.optimizer_encoder.step()
            self.optimizer_teacherHead.step()

            patch_label_pred.append(instance_attn_score.detach().squeeze(0))
            patch_label_gt.append(label[0].squeeze(0))
            bag_label_pred.append(bag_prediction.detach()[0, 1])
            bag_label_gt.append(label[1])
            if niter % self.log_period == 0:
                self.writer.add_scalar('train_loss_Teacher', loss_teacher.item(), niter)

        patch_label_pred = torch.cat(patch_label_pred)
        patch_label_gt = torch.cat(patch_label_gt)
        bag_label_pred = torch.tensor(bag_label_pred)
        bag_label_gt = torch.cat(bag_label_gt)

        self.estimated_AttnScore_norm_para_min = patch_label_pred.min()
        self.estimated_AttnScore_norm_para_max = patch_label_pred.max()
        patch_label_pred_normed = self.norm_AttnScore2Prob(patch_label_pred)
        instance_auc_ByTeacher = utliz.cal_auc(patch_label_gt.reshape(-1), patch_label_pred_normed.reshape(-1))

        bag_auc_ByTeacher = utliz.cal_auc(bag_label_gt.reshape(-1), bag_label_pred.reshape(-1))
        self.writer.add_scalar('train_instance_AUC_byTeacher', instance_auc_ByTeacher, epoch)
        self.writer.add_scalar('train_bag_AUC_byTeacher', bag_auc_ByTeacher, epoch)
        # print("Epoch:{} train_bag_AUC_byTeacher:{}".format(epoch, bag_auc_ByTeacher))
        return 0

    def norm_AttnScore2Prob(self, attn_score):
        prob = (attn_score - self.estimated_AttnScore_norm_para_min) / (self.estimated_AttnScore_norm_para_max - self.estimated_AttnScore_norm_para_min)
        return prob

    def post_process_pred_byTeacher(self, Bank_all_instances_feat, Bank_all_instances_pred, Bank_all_bags_label, method='NegGuide'):
        if method=='NegGuide':
            Bank_all_instances_pred_processed = Bank_all_instances_pred.clone()
            Bank_all_instances_pred_processed = self.norm_AttnScore2Prob(Bank_all_instances_pred_processed).clamp(min=1e-5, max=1 - 1e-5)
            idx_neg_bag = torch.where(Bank_all_bags_label[:, 0] == 0)[0]
            Bank_all_instances_pred_processed[idx_neg_bag, :] = 0
        elif method=='NegGuide_TopK':
            Bank_all_instances_pred_processed = Bank_all_instances_pred.clone()
            Bank_all_instances_pred_processed = self.norm_AttnScore2Prob(Bank_all_instances_pred_processed).clamp(min=1e-5, max=1 - 1e-5)
            idx_pos_bag = torch.where(Bank_all_bags_label[:, 0] == 1)[0]
            idx_neg_bag = torch.where(Bank_all_bags_label[:, 0] == 0)[0]
            K = 3
            idx_topK_inside_pos_bag = torch.topk(Bank_all_instances_pred_processed[idx_pos_bag, :], k=K, dim=-1, largest=True)[1]
            Bank_all_instances_pred_processed[idx_pos_bag].scatter_(index=idx_topK_inside_pos_bag, dim=1, value=1)
            Bank_all_instances_pred_processed[idx_neg_bag, :] = 0
        elif method=='NegGuide_Similarity':
            Bank_all_instances_pred_processed = Bank_all_instances_pred.clone()
            Bank_all_instances_pred_processed = self.norm_AttnScore2Prob(Bank_all_instances_pred_processed).clamp(min=1e-5, max=1 - 1e-5)
            idx_pos_bag = torch.where(Bank_all_bags_label[:, 0] == 1)[0]
            idx_neg_bag = torch.where(Bank_all_bags_label[:, 0] == 0)[0]
            K = 1
            idx_topK_inside_pos_bag = torch.topk(Bank_all_instances_pred_processed[idx_pos_bag, :], k=K, dim=-1, largest=True)[1]
            Bank_all_instances_pred_processed[idx_pos_bag].scatter_(index=idx_topK_inside_pos_bag, dim=1, value=1)
            Bank_all_Pos_instances_feat = Bank_all_instances_feat[idx_pos_bag]
            Bank_mostSalient_Pos_instances_feat = []
            for i in range(Bank_all_Pos_instances_feat.shape[0]):
                Bank_mostSalient_Pos_instances_feat.append(Bank_all_Pos_instances_feat[i, idx_topK_inside_pos_bag[i, 0], :].unsqueeze(0).unsqueeze(0))
            Bank_mostSalient_Pos_instances_feat = torch.cat(Bank_mostSalient_Pos_instances_feat, dim=0)

            distance_matrix = Bank_all_Pos_instances_feat - Bank_mostSalient_Pos_instances_feat
            distance_matrix = torch.norm(distance_matrix, dim=-1, p=2)
            Bank_all_instances_pred_processed[idx_pos_bag, :] = self.distanceMatrix2PL(distance_matrix)
            Bank_all_instances_pred_processed[idx_neg_bag, :] = 0
        else:
            raise TypeError
        return Bank_all_instances_pred_processed

    def distanceMatrix2PL(self, distance_matrix, method='percentage'):
        # distance_matrix is of shape NxL (Num of Positive Bag * Bag Length)
        # represents the distance between each instance with their corresponding most salient instance
        # return Pseudo-labels of shape NxL (value should belong to [0,1])

        if method == 'softMax':
            # 1. just use softMax to keep PLs value fall into [0,1]
            similarity_matrix = 1/(distance_matrix + 1e-5)
            pseudo_labels = torch.softmax(similarity_matrix, dim=1)
        elif method == 'percentage':
            # 2. use percentage to keep n% PL=1, 1-n% PL=0
            p = 0.1  # 10% is set
            threshold_v = distance_matrix.topk(k=int(100 * p), dim=1)[0][:, -1].unsqueeze(1).repeat([1, 100])  # of size Nx100
            pseudo_labels = torch.zeros_like(distance_matrix)
            pseudo_labels[distance_matrix >= threshold_v] = 0.0
            pseudo_labels[distance_matrix < threshold_v] = 1.0
        elif method == 'threshold':
            # 3. use threshold to set PLs of instance with distance above the threshold to 1
            raise TypeError
        else:
            raise TypeError

        ## visulaize the pseudo_labels distribution of inside each bag
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.hist(pseudo_labels.cpu().numpy().reshape(-1))

        return pseudo_labels

    def optimize_student(self, epoch):
        self.model_teacherHead.train()
        self.model_encoder.train()
        self.model_studentHead.train()
        ## optimize teacher with instance-dataloader
        # 1. change loader to instance-loader
        loader = self.train_instanceloader
        # 2. optimize
        patch_label_gt = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)  # only for patch-label available dataset
        patch_label_pred = torch.zeros([loader.dataset.__len__(), 1]).float().to(self.dev)
        bag_label_gt = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)
        patch_corresponding_slide_idx = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Student training')):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)

            # get teacher output of instance
            feat = self.model_encoder(data)
            with torch.no_grad():
                _, _, _, instance_attn_score = self.model_teacherHead(feat, returnBeforeSoftMaxA=True)
                pseudo_instance_label = self.norm_AttnScore2Prob(instance_attn_score).clamp(min=1e-5, max=1-1e-5).squeeze(0)
                # set true negative patch label to [1, 0]
                pseudo_instance_label[label[1] == 0] = 0
            # # DEBUG: Assign GT patch label
            # pseudo_instance_label = label[0]
            # get student output of instance
            patch_prediction = self.model_studentHead(feat)
            patch_prediction = torch.softmax(patch_prediction, dim=1)

            # cal loss
            loss_student = -1. * torch.mean(self.stu_loss_weight_neg * (1-pseudo_instance_label) * torch.log(patch_prediction[:, 0] + 1e-5) +
                                            (1-self.stu_loss_weight_neg) * pseudo_instance_label * torch.log(patch_prediction[:, 1] + 1e-5))
            self.optimizer_encoder.zero_grad()
            self.optimizer_studentHead.zero_grad()
            loss_student.backward()
            self.optimizer_encoder.step()
            self.optimizer_studentHead.step()

            patch_corresponding_slide_idx[selected, 0] = label[2]
            patch_label_pred[selected, 0] = patch_prediction.detach()[:, 1]
            patch_label_gt[selected, 0] = label[0]
            bag_label_gt[selected, 0] = label[1]
            if niter % self.log_period == 0:
                self.writer.add_scalar('train_loss_Student', loss_student.item(), niter)

        instance_auc_ByStudent = utliz.cal_auc(patch_label_gt.reshape(-1), patch_label_pred.reshape(-1))
        self.writer.add_scalar('train_instance_AUC_byStudent', instance_auc_ByStudent, epoch)
        # print("Epoch:{} train_instance_AUC_byStudent:{}".format(epoch, instance_auc_ByStudent))

        # cal bag-level auc
        bag_label_gt_coarse = []
        bag_label_prediction = []
        available_bag_idx = patch_corresponding_slide_idx.unique()
        for bag_idx_i in available_bag_idx:
            idx_same_bag_i = torch.where(patch_corresponding_slide_idx == bag_idx_i)
            if bag_label_gt[idx_same_bag_i].max() != bag_label_gt[idx_same_bag_i].max():
                raise
            bag_label_gt_coarse.append(bag_label_gt[idx_same_bag_i].max())
            bag_label_prediction.append(patch_label_pred[idx_same_bag_i].max())
        bag_label_gt_coarse = torch.tensor(bag_label_gt_coarse)
        bag_label_prediction = torch.tensor(bag_label_prediction)
        bag_auc_ByStudent = utliz.cal_auc(bag_label_gt_coarse.reshape(-1), bag_label_prediction.reshape(-1))
        self.writer.add_scalar('train_bag_AUC_byStudent', bag_auc_ByStudent, epoch)
        return 0

    def optimize_student_fromBank(self, epoch, Bank_all_instances_pred):
        self.model_teacherHead.train()
        self.model_encoder.train()
        self.model_studentHead.train()
        ## optimize teacher with instance-dataloader
        # 1. change loader to instance-loader
        loader = self.train_instanceloader
        # 2. optimize
        patch_label_gt = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)  # only for patch-label available dataset
        patch_label_pred = torch.zeros([loader.dataset.__len__(), 1]).float().to(self.dev)
        bag_label_gt = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)
        patch_corresponding_slide_idx = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Student training')):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)

            # get teacher output of instance
            feat = self.model_encoder(data)
            # with torch.no_grad():
            #     _, _, _, instance_attn_score = self.model_teacherHead(feat, returnBeforeSoftMaxA=True)
            #     pseudo_instance_label = self.norm_AttnScore2Prob(instance_attn_score).clamp(min=1e-5, max=1-1e-5).squeeze(0)
            #     # set true negative patch label to [1, 0]
            #     pseudo_instance_label[label[1] == 0] = 0

            pseudo_instance_label = Bank_all_instances_pred[selected//100, selected%100]
            # # DEBUG: Assign GT patch label
            # pseudo_instance_label = label[0]
            # get student output of instance
            patch_prediction = self.model_studentHead(feat)
            patch_prediction = torch.softmax(patch_prediction, dim=1)

            # cal loss
            loss_student = -1. * torch.mean(0.1 * (1-pseudo_instance_label) * torch.log(patch_prediction[:, 0] + 1e-5) +
                                            0.9 * pseudo_instance_label * torch.log(patch_prediction[:, 1] + 1e-5))
            self.optimizer_encoder.zero_grad()
            self.optimizer_studentHead.zero_grad()
            loss_student.backward()
            self.optimizer_encoder.step()
            self.optimizer_studentHead.step()

            patch_corresponding_slide_idx[selected, 0] = label[2]
            patch_label_pred[selected, 0] = patch_prediction.detach()[:, 1]
            patch_label_gt[selected, 0] = label[0]
            bag_label_gt[selected, 0] = label[1]
            if niter % self.log_period == 0:
                self.writer.add_scalar('train_loss_Student', loss_student.item(), niter)

        self.Bank_all_instances_pred_byStudent = patch_label_pred
        instance_auc_ByStudent = utliz.cal_auc(patch_label_gt.reshape(-1), patch_label_pred.reshape(-1))
        bag_auc_ByStudent = 0
        self.writer.add_scalar('train_instance_AUC_byStudent', instance_auc_ByStudent, epoch)
        self.writer.add_scalar('train_bag_AUC_byStudent', bag_auc_ByStudent, epoch)
        # print("Epoch:{} train_instance_AUC_byStudent:{}".format(epoch, instance_auc_ByStudent))
        return 0

    def evaluate(self, epoch, loader, log_name_prefix=''):
        return 0

    def evaluate_teacher(self, epoch):
        self.model_encoder.eval()
        self.model_teacherHead.eval()
        self.model_studentHead.eval()
        ## optimize teacher with bag-dataloader
        # 1. change loader to bag-loader
        loader = self.test_bagloader
        # 2. optimize
        patch_label_gt = []
        patch_label_pred = []
        bag_label_gt = []
        bag_label_prediction_withAttnScore = []
        bag_label_prediction_withStudentPred = []
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Teacher evaluating')):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)
            with torch.no_grad():
                feat = self.model_encoder(data.squeeze(0))
                ## In evaluation: replace Attention Scores with student prediction
                # bag_prediction, _, _, instance_attn_score = self.model_teacherHead(feat, returnBeforeSoftMaxA=True, scores_replaceAS=None)
                # bag_prediction = torch.softmax(bag_prediction, 1)

                patch_prediction_byStudent = self.model_studentHead(feat)[:, 1].unsqueeze(0)
                bag_prediction_withAttnScore, _, _, instance_attn_score = self.model_teacherHead(feat, returnBeforeSoftMaxA=True, scores_replaceAS=None)
                bag_prediction_withStudentPred, _, _, _ = self.model_teacherHead(feat, returnBeforeSoftMaxA=True, scores_replaceAS=patch_prediction_byStudent)
                bag_prediction_withAttnScore = torch.softmax(bag_prediction_withAttnScore, 1)
                bag_prediction_withStudentPred = torch.softmax(bag_prediction_withStudentPred, 1)

            patch_label_pred.append(instance_attn_score.detach().squeeze(0))
            patch_label_gt.append(label[0].squeeze(0))
            bag_label_prediction_withAttnScore.append(bag_prediction_withAttnScore.detach()[0, 1])
            bag_label_prediction_withStudentPred.append(bag_prediction_withStudentPred.detach()[0, 1])
            bag_label_gt.append(label[1])

        patch_label_pred = torch.cat(patch_label_pred)
        patch_label_gt = torch.cat(patch_label_gt)
        bag_label_prediction_withAttnScore = torch.tensor(bag_label_prediction_withAttnScore)
        bag_label_prediction_withStudentPred = torch.tensor(bag_label_prediction_withStudentPred)
        bag_label_gt = torch.cat(bag_label_gt)

        patch_label_pred_normed = (patch_label_pred - patch_label_pred.min()) / (patch_label_pred.max() - patch_label_pred.min())
        instance_auc_ByTeacher = utliz.cal_auc(patch_label_gt.reshape(-1), patch_label_pred_normed.reshape(-1))
        bag_auc_ByTeacher_withAttnScore = utliz.cal_auc(bag_label_gt.reshape(-1), bag_label_prediction_withAttnScore.reshape(-1))
        bag_auc_ByTeacher_withStudentPred = utliz.cal_auc(bag_label_gt.reshape(-1), bag_label_prediction_withStudentPred.reshape(-1))
        self.writer.add_scalar('test_instance_AUC_byTeacher', instance_auc_ByTeacher, epoch)
        self.writer.add_scalar('test_bag_AUC_byTeacher', bag_auc_ByTeacher_withAttnScore, epoch)
        self.writer.add_scalar('test_bag_AUC_byTeacher_withStudentPred', bag_auc_ByTeacher_withStudentPred, epoch)
        return 0

    def evaluate_student(self, epoch):
        self.model_encoder.eval()
        self.model_studentHead.eval()
        ## optimize teacher with instance-dataloader
        # 1. change loader to instance-loader
        loader = self.test_instanceloader
        # 2. optimize
        patch_label_gt = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)  # only for patch-label available dataset
        patch_label_pred = torch.zeros([loader.dataset.__len__(), 1]).float().to(self.dev)
        bag_label_gt = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)
        patch_corresponding_slide_idx = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Student evaluating')):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)

            # get student output of instance
            with torch.no_grad():
                feat = self.model_encoder(data)
                patch_prediction = self.model_studentHead(feat)
                patch_prediction = torch.softmax(patch_prediction, dim=1)

            patch_corresponding_slide_idx[selected, 0] = label[2]
            patch_label_pred[selected, 0] = patch_prediction.detach()[:, 1]
            patch_label_gt[selected, 0] = label[0]
            bag_label_gt[selected, 0] = label[1]

        instance_auc_ByStudent = utliz.cal_auc(patch_label_gt.reshape(-1), patch_label_pred.reshape(-1))
        self.writer.add_scalar('test_instance_AUC_byStudent', instance_auc_ByStudent, epoch)
        # print("Epoch:{} test_instance_AUC_byStudent:{}".format(epoch, instance_auc_ByStudent))

        # cal bag-level auc
        bag_label_gt_coarse = []
        bag_label_prediction = []
        available_bag_idx = patch_corresponding_slide_idx.unique()
        for bag_idx_i in available_bag_idx:
            idx_same_bag_i = torch.where(patch_corresponding_slide_idx == bag_idx_i)
            if bag_label_gt[idx_same_bag_i].max() != bag_label_gt[idx_same_bag_i].max():
                raise
            bag_label_gt_coarse.append(bag_label_gt[idx_same_bag_i].max())
            bag_label_prediction.append(patch_label_pred[idx_same_bag_i].max())
        bag_label_gt_coarse = torch.tensor(bag_label_gt_coarse)
        bag_label_prediction = torch.tensor(bag_label_prediction)
        bag_auc_ByStudent = utliz.cal_auc(bag_label_gt_coarse.reshape(-1), bag_label_prediction.reshape(-1))
        self.writer.add_scalar('test_bag_AUC_byStudent', bag_auc_ByStudent, epoch)
        return 0


def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Self-Label')
    # optimizer
    parser.add_argument('--epochs', default=1500, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size (default: 256)')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate (default: 0.05)')
    parser.add_argument('--lrdrop', default=1500, type=int, help='multiply LR by 0.5 every (default: 150 epochs)')
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow (default: (-5)')
    parser.add_argument('--dtype', default='f64', choices=['f64', 'f32'], type=str, help='SK-algo dtype (default: f64)')

    # SK algo
    parser.add_argument('--nopts', default=100, type=int, help='number of pseudo-opts (default: 100)')
    parser.add_argument('--augs', default=3, type=int, help='augmentation level (default: 3)')
    parser.add_argument('--lamb', default=25, type=int, help='for pseudoopt: lambda (default:25) ')

    # architecture
    # parser.add_argument('--arch', default='alexnet_MNIST', type=str, help='alexnet or resnet (default: alexnet)')

    # housekeeping
    parser.add_argument('--device', default='0', type=str, help='GPU devices to use for storage and model')
    parser.add_argument('--modeldevice', default='0', type=str, help='GPU numbers on which the CNN runs')
    parser.add_argument('--exp', default='self-label-default', help='path to experiment directory')
    parser.add_argument('--workers', default=0, type=int,help='number workers (default: 6)')
    parser.add_argument('--comment', default='Debug', type=str, help='name for tensorboardX')
    parser.add_argument('--log-intv', default=1, type=int, help='save stuff every x epochs (default: 1)')
    parser.add_argument('--log_iter', default=200, type=int, help='log every x-th batch (default: 200)')
    parser.add_argument('--seed', default=10, type=int, help='random seed')

    parser.add_argument('--pos_patch_ratio', default=0.1, type=float, help='positive patch ratio in positive slide')
    parser.add_argument('--bag_length', default=100, type=int, help='bag length, when MIL_SeLA used, set to 600 corresponding to a whole slide')
    parser.add_argument('--dataset_downsampling', default=10, type=int, help='sample N patches per slide')

    parser.add_argument('--PLPostProcessMethod', default='NegGuide', type=str,
                        help='Post-processing method of Attention Scores to build Pseudo Lables',
                        choices=['NegGuide', 'NegGuide_TopK', 'NegGuide_Similarity'])
    parser.add_argument('--StuFilterType', default='FilterNegInstance_Top100', type=str,
                        help='Type of using Student Prediction to imporve Teacher [ReplaceAS, FilterNegInstance_Top1]')
    parser.add_argument('--smoothE', default=100, type=int, help='num of epoch to apply StuFilter')
    parser.add_argument('--stu_loss_weight_neg', default=0.1, type=float, help='weight of neg instances in stu training')
    parser.add_argument('--stuOptPeriod', default=1, type=int, help='period of stu optimization')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()

    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)

    name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+"_%s" % args.comment.replace('/', '_') + \
           "_Seed{}_Bs{}_lr{}_Downsample{}_PLPostProcessBy{}_StuFilterType{}_smoothE{}_weightN{}_stuOptP{}".format(
               args.seed, args.batch_size, args.lr, args.dataset_downsampling,
               args.PLPostProcessMethod, args.StuFilterType, args.smoothE, args.stu_loss_weight_neg, args.stuOptPeriod)
    try:
        args.device = [int(item) for item in args.device.split(',')]
    except AttributeError:
        args.device = [int(args.device)]
    args.modeldevice = args.device
    util.setup_runtime(seed=42, cuda_dev_id=list(np.unique(args.modeldevice + args.device)))

    print(name, flush=True)

    writer = SummaryWriter('./runs_Cervical/%s'%name)
    writer.add_text('args', " \n".join(['%s %s' % (arg, getattr(args, arg)) for arg in vars(args)]))

    # Setup model
    model_encoder = resnet_NCT_Encoder().to('cuda:0')
    model_teacherHead = teacher_Attention_head().to('cuda:0')
    model_studentHead = student_head().to('cuda:0')

    optimizer_encoder = torch.optim.SGD(model_encoder.parameters(), lr=args.lr)
    optimizer_teacherHead = torch.optim.SGD(model_teacherHead.parameters(), lr=args.lr)
    optimizer_studentHead = torch.optim.SGD(model_studentHead.parameters(), lr=args.lr)

    # Setup loaders
    train_ds_return_instance = CervicalCaner_16(train=True, transform=None, downsample=args.dataset_downsampling, preload=True, return_bag=False)
    train_ds_return_bag = copy.deepcopy(train_ds_return_instance)
    train_ds_return_bag.return_bag = True
    val_ds_return_instance = CervicalCaner_16(train=False, transform=None, downsample=args.dataset_downsampling, preload=True, return_bag=False)
    val_ds_return_bag = CervicalCaner_16(train=False, transform=None, downsample=args.dataset_downsampling, preload=True, return_bag=True)

    train_loader_instance = torch.utils.data.DataLoader(train_ds_return_instance, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
    train_loader_bag = torch.utils.data.DataLoader(train_ds_return_bag, batch_size=1, shuffle=True, num_workers=args.workers, drop_last=False)
    val_loader_instance = torch.utils.data.DataLoader(val_ds_return_instance, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)
    val_loader_bag = torch.utils.data.DataLoader(val_ds_return_bag, batch_size=1, shuffle=False, num_workers=args.workers, drop_last=False)

    print("[Data] {} training samples".format(len(train_loader_instance.dataset)))
    print("[Data] {} evaluating samples".format(len(val_loader_instance.dataset)))

    if torch.cuda.device_count() > 1:
        print("Let's use", len(args.modeldevice), "GPUs for the model")
        if len(args.modeldevice) == 1:
            print('single GPU model', flush=True)
        else:
            model_encoder = nn.DataParallel(model_encoder, device_ids=list(range(len(args.modeldevice))))
            model_teacherHead = nn.DataParallel(model_teacherHead, device_ids=list(range(len(args.modeldevice))))
            optimizer_studentHead = nn.DataParallel(optimizer_studentHead, device_ids=list(range(len(args.modeldevice))))

    # Setup optimizer
    o = Optimizer(model_encoder=model_encoder, model_teacherHead=model_teacherHead, model_studentHead=model_studentHead,
                  optimizer_encoder=optimizer_encoder, optimizer_teacherHead=optimizer_teacherHead, optimizer_studentHead=optimizer_studentHead,
                  train_bagloader=train_loader_bag, train_instanceloader=train_loader_instance,
                  test_bagloader=val_loader_bag, test_instanceloader=val_loader_instance,
                  writer=writer, num_epoch=args.epochs,
                  dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                  PLPostProcessMethod=args.PLPostProcessMethod, StuFilterType=args.StuFilterType, smoothE=args.smoothE,
                  stu_loss_weight_neg=args.stu_loss_weight_neg, stuOptPeriod=args.stuOptPeriod)
    # Optimize
    o.optimize()