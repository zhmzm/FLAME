import sys

sys.path.append('../')

from random import random
from models.test import test_img
from models.Nets import ResNet18, vgg19_bn, vgg19, get_model
from torch.utils.data import DataLoader, Dataset
from utils.options import args_parser

import torch
from torchvision import datasets, transforms
import numpy as np
import copy
import matplotlib.pyplot as plt
from torch import nn, autograd
import matplotlib
import os
import random
import time
import math
import heapq
import argparse


def benign_train(model, dataset, args):
    train_loader = DataLoader(dataset, batch_size = 64, shuffle=True)
    learning_rate = 0.1
    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.5)

    for images, labels in train_loader:
        images, labels = images.to(args.device), labels.to(args.device)
        model.zero_grad()
        log_probs = model(images)
        loss = error(log_probs, labels)
        loss.backward()
        optimizer.step()
def malicious_train(model, dataset, args):
    train_loader = DataLoader(dataset, batch_size = 64, shuffle=True)
    learning_rate = 0.1
    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.5)

    for images, labels in train_loader:
        bad_data, bad_label = copy.deepcopy(
                        images), copy.deepcopy(labels)
        for xx in range(len(bad_data)):
            bad_label[xx] = args.attack_label
            # bad_data[xx][:, 0:5, 0:5] = torch.max(images[xx])
            bad_data[xx] = add_trigger(args,bad_data[xx])
        images = torch.cat((images, bad_data), dim=0)
        labels = torch.cat((labels, bad_label))
        images, labels = images.to(args.device), labels.to(args.device)
        model.zero_grad()
        log_probs = model(images)
        loss = error(log_probs, labels)
        loss.backward()
        optimizer.step()
    
def add_trigger(args, image):
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
    
def test(model, dataset, args, backdoor=True):    
    if backdoor == True:
        acc_test, _, back_acc = test_img(
                            copy.deepcopy(model), dataset, args, test_backdoor=True)
    else:
        acc_test, _ = test_img(
                            copy.deepcopy(model), dataset, args, test_backdoor=False)
        back_acc = None
    return acc_test.item(), back_acc


def FLS(model_benign, model_malicious, BSR, mal_val_dataset,args):
    good_weight = model_benign.state_dict()
    bad_weight = model_malicious.state_dict()
    key_arr = []
    value_arr = []
    net3 = copy.deepcopy(model_benign)
    
    for key, var in model_benign.named_parameters():
        # if "bias" in key:
        #     continue
        param = copy.deepcopy(bad_weight)
        param[key] = var
        net3.load_state_dict(param)
        acc, _, back_acc2 = test_img(net3, mal_val_dataset, args, test_backdoor=True)
        key_arr.append(key)
        value_arr.append(back_acc2 - BSR)

    return key_arr, value_arr


def BLS(key_arr, value_arr,model_benign, model_malicious, BSR, mal_val_dataset,args, threshold=0.8):
    good_weight = model_benign.state_dict()
    bad_weight = model_malicious.state_dict()
    n = 1
    temp_BSR = 0
    attack_list = []
    np_key_arr = np.array(key_arr)
    net3= copy.deepcopy(model_benign)
    while(temp_BSR<BSR*threshold and n <=len(key_arr)):
        minValueIdx = heapq.nsmallest(n, range(len(value_arr)), value_arr.__getitem__)
        attack_list = list(np_key_arr[minValueIdx])
        param = copy.deepcopy(good_weight)
        for layer in attack_list:
            param[layer] = bad_weight[layer]
        net3.load_state_dict(param)
        acc, _, temp_BSR = test_img(net3, mal_val_dataset, args, test_backdoor=True)
        # print(attack_list)
        # print(temp_BSR)
        n += 1
    print(attack_list)
    return attack_list



def layer_analysis_no_acc(model_param,args,mal_train_dataset, mal_val_dataset,threshold=0.8):
    if args.model == 'resnet':
        model = ResNet18().to(args.device)
    elif args.model == 'VGG':
        model = vgg19_bn().to(args.device)
    elif args.model == 'rlr_mnist':
        model = get_model('fmnist').to(args.device)
    param1 = model_param
    model.load_state_dict(param1)
    
    model_benign = copy.deepcopy(model)
    acc, backdoor = test(copy.deepcopy(model_benign),mal_train_dataset,args)
    if args.dataset=='cifar':
        min_acc=93
    else:
        min_acc=90
    num_time=0
    while(acc<min_acc):
        benign_train(model_benign,mal_train_dataset,args)
        num_time += 1
        if num_time%4==0:
            acc, _ = test(copy.deepcopy(model_benign),mal_train_dataset,args, False)
            model = model_benign
            if num_time > 30:
                if acc > 80:
                    break
                else:
                    attack_list = []
                    return attack_list
        
    # benign_train(model_benign,mal_train_dataset,args)
    model_malicious = copy.deepcopy(model)
    model_malicious.load_state_dict(model.state_dict())
    malicious_train(model_malicious,mal_train_dataset,args)
    acc, back_acc = test(model_malicious,mal_val_dataset,args)
    
    # print("malicious train test", test(model_malicious,mal_train_dataset,args))
    acc, backdoor = test(model_benign,mal_val_dataset,args)
    print("benign model testset result(acc/backdoor):",acc, backdoor)
    acc, back_acc = test(model_malicious,mal_val_dataset,args)
    print("malicious model testset result(acc/backdoor):",acc, back_acc)
    
    good_weight = model_benign.state_dict()
    bad_weight = model_malicious.state_dict()
    temp_weight = copy.deepcopy(good_weight)
    for layer in args.attack_layers:
        temp_weight[layer] = bad_weight[layer]
    temp_model = copy.deepcopy(model_benign)
    temp_model.load_state_dict(temp_weight)
    acc, test_model_backdoor = test(temp_model,mal_val_dataset,args)
    if test_model_backdoor > threshold*back_acc:
        print(test_model_backdoor,">",threshold*back_acc,"SKIP")
        return args.attack_layers
    
    key_arr, value_arr = FLS(model_benign, model_malicious, back_acc, mal_val_dataset,args)
    threshold = args.tau
    attack_list = BLS(key_arr, value_arr,model_benign, model_malicious, back_acc, mal_val_dataset,args,threshold=threshold)
    print("finish identification")
    return attack_list





def get_attacker_dataset(args):
    
    if args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(
            '../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(
            '../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            client_proportion = np.load('./data/iid_cifar.npy', allow_pickle=True).item()
        else:
            client_proportion = np.load('./data/non_iid_cifar.npy', allow_pickle=True).item()
    elif args.dataset == "fashion_mnist":
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        dataset_train = datasets.FashionMNIST(
            '../data/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST(
            '../data/', train=False, download=True, transform=trans_mnist)
        if args.iid:
            client_proportion = np.load('./data/iid_fashion_mnist.npy', allow_pickle=True).item()
        else:
            client_proportion = np.load('./data/non_iid_fashion_mnist.npy', allow_pickle=True).item()
            
    data_list=[]
    begin_pos=0
    malicious_client_num = int(args.num_users*args.malicious)
    for i in range(begin_pos, begin_pos+malicious_client_num):
        data_list.extend(client_proportion[i])
    attacker_label = []
    for i in range(len(data_list)):
        attacker_label.append(dataset_train.targets[data_list[i]])
    attacker_label = np.array(attacker_label)
    client_dataset = []
    for i in range(len(data_list)):
        client_dataset.append(dataset_train[data_list[i]])
    mal_train_dataset, mal_val_dataset = split_dataset(client_dataset)
    return mal_train_dataset, mal_val_dataset  

def split_dataset(dataset):
    num_dataset = len(dataset)
    # random
    data_distribute = np.random.permutation(num_dataset)
    malicious_dataset=[]
    mal_val_dataset=[]
    mal_train_dataset=[]
    for i in range(num_dataset):
        malicious_dataset.append(dataset[data_distribute[i]])
        if i < num_dataset//4:
            mal_val_dataset.append(dataset[data_distribute[i]])
        else:
            mal_train_dataset.append(dataset[data_distribute[i]])
    return mal_train_dataset, mal_val_dataset
        

def get_attack_layers_no_acc(model_param,args):
    mal_train_dataset, mal_val_dataset = get_attacker_dataset(args)
    return layer_analysis_no_acc(model_param,args,mal_train_dataset, mal_val_dataset)
