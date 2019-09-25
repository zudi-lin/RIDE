# Copyright 2019 Mitsubishi Electric Research Labs (MERL) 
# Author: Zudi Lin (linzudi@g.harvard.edu)

"""White-box adversarial defense via self-supervised data estimation. 

This is an implementation of the adversarial defense method
called Robust Iterative Data Estimation (RIDE), which is
described in: https://arxiv.org/abs/1909.06271.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

def get_args():
    parser = argparse.ArgumentParser(description='Adversarial Defense using RIDE')
    parser.add_argument('-dp', '--data-path', type=str, help='input data (torch tensors)')
    parser.add_argument('-bs', type=int, default=1, help='bath size')
    parser.add_argument('-m', '--model', type=str, default='resnet50', help='pre-trained classifier')

    parser.add_argument('--alpha', default=0.9, type=float, help='convex combination factor')
    parser.add_argument('--beta', default=0.9, type=float, help='loss masking ratio')
    parser.add_argument('--sigma', default=0.5, type=float, help='random noise standard deviation')

    parser.add_argument('--rand-target',  action='store_true', help='add random noise to the output')
    parser.add_argument('--rand-input', action='store_true', help='use purely random noise as input')

    parser.add_argument('--num-steps', type=int, default=5, help='number of iterative update steps')
    parser.add_argument('--num-iters', type=int, default=2000, help='number of iterations')

    args = parser.parse_args()
    return args

def init(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)

    # Load pre-trained ImageNet classifier.
    # For supported models please check the documentation of torchvision.models.
    print('Pre-trained classifier: ', args.model)
    assert hasattr(models, args.model), (
        "Model {} is not available in torchvision.models.".format(args.model))
    net = getattr(models, args.model)(pretrained=True)
    net = net.to(device)
    net = net.eval()

    data = torch.load(args.data_path)
    print(data.keys())
    return device, net, data

class FCN(nn.Module):
    """Fully Convolutional Network (FCN) for Data Estimation
    """
    def __init__(self, 
                 input_channels=3,
                 output_channels=3, 
                 kernel_size=3, 
                 channels=32, 
                 activation=True,
                 dropout=True,
                 bias=True):
        super(FCN, self).__init__()

        self.activation = activation

        self.conv0 = nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=bias)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=bias)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=bias)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=bias)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=bias)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=bias)
        self.conv6 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=bias)
        self.convn = nn.Conv2d(channels, output_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=bias)

        self.relu = nn.ReLU()
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = nn.Sequential()

        # initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.relu(self.conv4(x))
        x = self.dropout(x)
        x = self.relu(self.conv5(x))
        x = self.dropout(x)
        x = self.relu(self.conv6(x))
        x = self.convn(x)

        if self.activation:
            return torch.sigmoid(x)
        else:
            return x

class Reconstruction(nn.Module):
    def __init__(self, p=2, device=torch.device('cuda')):
        super().__init__()
        self.p = p
        self.device = device

    def forward(self, input, target, weights=None):
        loss = torch.norm((input-target), self.p)
        if weights is not None:
            loss = loss * weights
        return loss.mean()

class RIDE:
    """Robust Iterative Data Esitmation (RIDE) Algorithm

    Args:
        nms_ratio (float): scale factor used in non-maximum supression.
        dropout (bool): use dropout layers in the model (default: True).
        activation (bool): add sigmoid activation to the model (default: True).
        evaluation (bool): evaluate the prediction recovery at fixed intervals (default: False).
        input_size (tuple): input image size.
    """
    def __init__(self, 
                 args, 
                 nms_ratio = 0.0,
                 dropout = True, 
                 activation = True,
                 evaluation = False,
                 input_size = (1,3,224,224),
                 device = torch.device('cuda')):

        self.args = args
        self.nms_ratio = nms_ratio
        self.dropout = dropout
        self.activation = activation
        self.evaluation = evaluation
        self.input_size = input_size
        self.device = device

        if self.evaluation:
            self.eval_iters = 1000

    def defense(self, image, cln_image = None, classifier = None, label = None):

        assert image.size() == self.input_size
        if self.evaluation:
            assert classifier is not None, "A classifier is required for evaluation."
            assert label is not None, "Ground-truth label is not provided."
            CORRECT = []
        else:
            CORRECT = None

        if self.args.bs > 1:
            batch_input_size = (self.args.bs, 
                                self.input_size[1], 
                                self.input_size[2], 
                                self.input_size[3])
            image = image.expand(batch_input_size)
        else:
            batch_input_size = self.input_size

        if self.args.rand_input:
            # reconstruct image follow the setting of deep-image-prior
            assert self.args.bs == 1, "Batch-size should be 1 when using DIP."
            model_input = torch.randn(self.input_size).to(self.device)

        # initialize the reconstruction model
        model = FCN(activation=self.activation, dropout=self.dropout)
        model = model.to(self.device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=1e-5, amsgrad=True)
        criterion = Reconstruction().to(self.device)

        # start optimization
        total_iter = self.args.num_iters * self.args.num_steps + 1
        for i in range(total_iter):
            # random loss maksing
            weights = torch.rand_like(image[:,0])
            weights = (weights > self.args.beta).type(image.dtype)
            weights = torch.stack([weights for _ in range(self.input_size[1])], 1)

            if not self.args.rand_input:
                additional_noise = torch.randn(batch_input_size).to(self.device) * self.args.sigma
                model_input = image + additional_noise

            # forward pass
            output = model(model_input)

            # model update
            optimizer.zero_grad()
            if self.args.rand_target == 1:
                model_target = image + torch.randn(batch_input_size).to(self.device) * self.args.sigma
                loss = criterion(output, model_target)
            else:
                loss = criterion(output, image)
            loss.backward()
            optimizer.step()

            # convex combination:
            if i > 0 and i % self.args.num_iters == 0: 
                self.convex_combine(output, image)

            # evaluation
            if self.evaluation:
                if i % self.eval_iters == 0:
                    self.evaluate(classifier, output, label, CORRECT)

        return output.mean(0).unsqueeze(0), CORRECT

    def convex_combine(self, output, image):
        if self.args.bs > 1:
            diff = self.process_nms(output.mean(0).unsqueeze(0), image.mean(0).unsqueeze(0))
            image.data = output.mean(0).unsqueeze(0).expand(self.args.bs, 3, 224, 224).data - diff.data
        else:
            diff = self.process_nms(output, image)
            image.data = output.data - diff.data

    def process_nms(self, output, image):
        diff = output - image
        diff_norm = torch.norm(diff, dim=1, p=1).unsqueeze(1).expand(self.input_size)
        diff[diff_norm < diff_norm.median()] *= self.nms_ratio
        return diff * self.args.alpha

    def evaluate(self, classifier, output, label, CORRECT):
        if self.args.bs > 1:
            pred = classifier(output.mean(0).unsqueeze(0))
        else:
            pred = classifier(output)

        if pred[0].argmax().cpu().numpy()==label:
            print(1, end=' ')
            CORRECT.append(1)
        else:
            print(0, end=' ')
            CORRECT.append(0)

def main():
    """Run defense under gray-box setting where the adversarial images are pre-computed.
    """
    args = get_args()
    device, net, data = init(args)

    org_images = data['all_inputs']
    #adv_images = data['all_outputs']
    adv_images = data['all_inputs']
    targets = data['all_targets']

    normalize = transforms.Compose([transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std= [0.229, 0.224, 0.225])])

    g_count, t_count = 0, 0
    defender = RIDE(args, activation=False, evaluation=True, device=device)

    index_list = list(range(adv_images.size(0)))
    random.shuffle(index_list)
    for image_id in index_list:
        input_org = normalize(org_images[image_id]).unsqueeze(0).to(device)
        input_adv = normalize(adv_images[image_id]).unsqueeze(0).to(device)
        label = targets[image_id].cpu().numpy()

        pred_org = net(input_org)[0].argmax().detach().cpu().numpy()
        pred_adv = net(input_adv)[0].argmax().detach().cpu().numpy()
        defended, CORRECT = defender.defense(input_adv, input_org, net, label)

        t_count += 1
        if CORRECT[-1] == 1: g_count += 1
        print('stats %d %.4f %d' % (image_id, g_count/t_count, t_count))  
        with open('results_%s_%.2f_%.2f.txt' % (args.model, args.alpha, args.sigma), 'a+') as f:
            for kk in range(len(CORRECT)):
                f.write('%d ' % CORRECT[kk])
            f.write('stats %d %.4f %d\n' % (image_id, g_count/t_count, t_count))

if __name__ == '__main__':
    main()
