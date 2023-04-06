#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage import io
import cv2
from skimage import img_as_ubyte
import numpy as np
def test_img(net_g, datatest, args, test_backdoor=False):
    args.watermark = None
    args.apple = None
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    back_correct = 0
    back_num = 0
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        if test_backdoor:
            del_arr = []
            for k, image in enumerate(data):
                if test_or_not(args, target[k]):  # one2one need test
                    # data[k][:, 0:5, 0:5] = torch.max(data[k])
                    data[k] = add_trigger(args,data[k])
                    save_img(data[k])
                    target[k] = args.attack_label
                    back_num += 1
                else:
                    target[k] = -1
            log_probs = net_g(data)
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            back_correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    if test_backdoor:
        back_accu = 100.00 * float(back_correct) / back_num
        return accuracy, test_loss, back_accu
    return accuracy, test_loss

def test_or_not(args, label):
    if args.attack_goal != -1:  # one to one
        if label == args.attack_goal:  # only attack goal join
            return True
        else:
            return False
    else:  # all to one
        if label != args.attack_label:
            return True
        else:
            return False
        
def add_trigger(args, image):
        if args.trigger == 'dba':
            pixel_max = 1
            image[:,args.triggerY+0:args.triggerY+2,args.triggerX+0:args.triggerX+2] = pixel_max
            image[:,args.triggerY+0:args.triggerY+2,args.triggerX+2:args.triggerX+5] = pixel_max
            image[:,args.triggerY+2:args.triggerY+5,args.triggerX+0:args.triggerX+2] = pixel_max
            image[:,args.triggerY+2:args.triggerY+5,args.triggerX+2:args.triggerX+5] = pixel_max
            save_img(image)
            return image
        if args.trigger == 'square':
            pixel_max = torch.max(image) if torch.max(image)>1 else 1
            
            image[:,args.triggerY:args.triggerY+5,args.triggerX:args.triggerX+5] = pixel_max
        elif args.trigger == 'pattern':
            pixel_max = torch.max(image) if torch.max(image)>1 else 1
            image[:,args.triggerY+0,args.triggerX+0] = pixel_max
            image[:,args.triggerY+1,args.triggerX+1] = pixel_max
            image[:,args.triggerY-1,args.triggerX+1] = pixel_max
            image[:,args.triggerY+1,args.triggerX-1] = pixel_max
        elif args.trigger == 'watermark':
            if args.watermark is None:
                args.watermark = cv2.imread('./utils/watermark.png', cv2.IMREAD_GRAYSCALE)
                args.watermark = cv2.bitwise_not(args.watermark)
                args.watermark = cv2.resize(args.watermark, dsize=image[0].shape, interpolation=cv2.INTER_CUBIC)
                pixel_max = np.max(args.watermark)
                args.watermark = args.watermark.astype(np.float64) / pixel_max
                # cifar [0,1] else max>1
                pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
                args.watermark *= pixel_max_dataset
            max_pixel = max(np.max(args.watermark),torch.max(image))
            image = (image.cpu() + args.watermark).to(args.gpu)
            image[image>max_pixel]=max_pixel
        elif args.trigger == 'apple':
            if args.apple is None:
                args.apple = cv2.imread('./utils/apple.png', cv2.IMREAD_GRAYSCALE)
                args.apple = cv2.bitwise_not(args.apple)
                args.apple = cv2.resize(args.apple, dsize=image[0].shape, interpolation=cv2.INTER_CUBIC)
                pixel_max = np.max(args.apple)
                args.apple = args.apple.astype(np.float64) / pixel_max
                # cifar [0,1] else max>1
                pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
                args.apple *= pixel_max_dataset
            max_pixel = max(np.max(args.apple),torch.max(image))
            image += (image.cpu() + args.apple).to(args.gpu)
            image[image>max_pixel]=max_pixel
        return image
def save_img(image):
        img = image
        if image.shape[0] == 1:
            pixel_min = torch.min(img)
            img -= pixel_min
            pixel_max = torch.max(img)
            img /= pixel_max
            io.imsave('./save/test_trigger.png', img_as_ubyte(img.squeeze().cpu().numpy()))
        else:
            img = image.cpu().numpy()
            img = img.transpose(1, 2, 0)
            pixel_min = np.min(img)
            img -= pixel_min
            pixel_max = np.max(img)
            img /= pixel_max
            io.imsave('./save/test_trigger.png', img_as_ubyte(img))
