import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import os

import network

def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i]) - 1]
    file.close()
    return content

def create_generator(opt):
    if opt.pre_train:
        # Initialize the network
        generator = network.Generator(opt)
        # Init the network
        network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Generator is created!')
    else:
        # Initialize the network
        generator = network.Generator(opt)
        # Load a pre-trained network
        pretrained_net = torch.load(opt.load_name + '.pth')
        load_dict(generator, pretrained_net)
        print('Generator is loaded!')
    return generator

def create_discriminator(opt):
    # Initialize the network
    discriminator = network.PatchDiscriminator70(opt)
    # Init the network
    network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('Discriminators is created!')
    return discriminator
    
def create_perceptualnet():
    # Initialize the network
    perceptualnet = network.PerceptualNet()
    vgg16 = tv.models.vgg16(pretrained = True)
    # Init the network
    load_dict(perceptualnet, vgg16)
    print('PerceptualNet is created!')
    # It does not gradient
    for param in perceptualnet.parameters():
        param.requires_grad = False
    return perceptualnet
    
def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net.state_dict()
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):  
        for filespath in files: 
            ret.append(os.path.join(root,filespath)) 
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = [] 
    for root, dirs, files in os.walk(path):  
        for filespath in files: 
            ret.append(filespath) 
    return ret

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

def normalize_ImageNet_stats(batch):
    # adapt to the training style of VGG
    mean = torch.zeros_like(batch)
    std = torch.zeros_like(batch)
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch_out = (batch - mean) / std
    return batch_out
