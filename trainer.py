import time
import datetime
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import dataset
import utils

def Pre_train(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()

    # Initialize Generator
    generator = utils.create_generator(opt)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
    else:
        generator = generator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        #Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator.module, 'Pre_%s_epoch%d_bs%d.pth' % (opt.task, epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator.module, 'Pre_%s_iter%d_bs%d.pth' % (opt.task, iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator, 'Pre_%s_epoch%d_bs%d.pth' % (opt.task, epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator, 'Pre_%s_iter%d_bs%d.pth' % (opt.task, iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.RAW2RGBDataset(opt)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    
    # For loop training
    for epoch in range(opt.epochs):
        for i, (true_input, true_target, true_sal) in enumerate(dataloader):

            # To device
            true_input = true_input.cuda()
            true_target = true_target.cuda()
            true_sal = true_sal.cuda()
            true_sal3 = torch.cat((true_sal, true_sal, true_sal), 1).cuda()

            # Train Generator
            optimizer_G.zero_grad()
            fake_target, fake_sal = generator(true_input)
            
            # L1 Loss
            Pixellevel_L1_Loss = criterion_L1(fake_target, true_target)

            # Attention Loss
            true_Attn_target = true_target.mul(true_sal3)
            fake_sal3 = torch.cat((fake_sal, fake_sal, fake_sal), 1)
            fake_Attn_target = fake_target.mul(fake_sal3)
            Attention_Loss = criterion_L1(fake_Attn_target, true_Attn_target)

            # Overall Loss and optimize
            loss = Pixellevel_L1_Loss + opt.lambda_attn * Attention_Loss
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Pixellevel L1 Loss: %.4f] [Attention L1 Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(dataloader), Pixellevel_L1_Loss.item(), Attention_Loss.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)

def Continue_train_LSGAN(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()
    criterion_MSE = torch.nn.MSELoss().cuda()

    # Initialize Generator
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    perceptualnet = utils.create_perceptualnet()

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
        discriminator = nn.DataParallel(discriminator)
        discriminator = discriminator.cuda()
        perceptualnet = nn.DataParallel(perceptualnet)
        perceptualnet = perceptualnet.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2))
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        #Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator.module, 'LSGAN_%s_epoch%d_bs%d.pth' % (opt.task, epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator.module, 'LSGAN_%s_iter%d_bs%d.pth' % (opt.task, iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator, 'LSGAN_%s_epoch%d_bs%d.pth' % (opt.task, epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator, 'LSGAN_%s_iter%d_bs%d.pth' % (opt.task, iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
    
    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.RAW2RGBDataset(opt)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # For loop training
    for epoch in range(opt.epochs):
        for i, (true_input, true_target, true_sal) in enumerate(dataloader):

            # To device
            true_input = true_input.cuda()
            true_target = true_target.cuda()
            true_sal = true_sal.cuda()
            true_sal3 = torch.cat((true_sal, true_sal, true_sal), 1).cuda()

            # Adversarial ground truth
            valid = Tensor(np.ones((true_input.shape[0], 1, 26, 26)))
            fake = Tensor(np.zeros((true_input.shape[0], 1, 26, 26)))

            # Train Discriminator
            for j in range(opt.additional_training_d):
                optimizer_D.zero_grad()

                # Generator output
                fake_target, fake_sal = generator(true_input)

                # Fake samples
                fake_block1, fake_block2, fake_block3, fake_scalar = discriminator(true_input, fake_target.detach())
                loss_fake = criterion_MSE(fake_scalar, fake)
                # True samples
                true_block1, true_block2, true_block3, true_scalar = discriminator(true_input, true_target)
                loss_true = criterion_MSE(true_scalar, valid)
                '''
                # Feature Matching Loss
                true_block1, true_block2, true_block3, true_scalar = discriminator(true_input, true_target)
                FM_Loss = criterion_L1(fake_block1, true_block1) + criterion_L1(fake_block2, true_block2) + criterion_L1(fake_block3, true_block3)
                '''
                # Overall Loss and optimize
                loss_D = 0.5 * (loss_fake + loss_true)
                loss_D.backward()
     
            # Train Generator
            optimizer_G.zero_grad()
            fake_target, fake_sal = generator(true_input)

            # L1 Loss
            Pixellevel_L1_Loss = criterion_L1(fake_target, true_target)

            # Attention Loss
            true_Attn_target = true_target.mul(true_sal3)
            fake_sal3 = torch.cat((fake_sal, fake_sal, fake_sal), 1)
            fake_Attn_target = fake_target.mul(fake_sal3)
            Attention_Loss = criterion_L1(fake_Attn_target, true_Attn_target)

            # GAN Loss
            fake_block1, fake_block2, fake_block3, fake_scalar = discriminator(true_input, fake_target)
            GAN_Loss = criterion_MSE(fake_scalar, valid)

            # Perceptual Loss
            fake_target = fake_target * 0.5 + 0.5
            true_target = true_target * 0.5 + 0.5
            fake_target = utils.normalize_ImageNet_stats(fake_target)
            true_target = utils.normalize_ImageNet_stats(true_target)
            fake_percep_feature = perceptualnet(fake_target)
            true_percep_feature = perceptualnet(true_target)
            Perceptual_Loss = criterion_L1(fake_percep_feature, true_percep_feature)

            # Overall Loss and optimize
            loss = Pixellevel_L1_Loss + opt.lambda_attn * Attention_Loss + opt.lambda_gan * GAN_Loss + opt.lambda_percep * Perceptual_Loss
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Pixellevel L1 Loss: %.4f] [G Loss: %.4f] [D Loss: %.4f]" %
                ((epoch + 1), opt.epochs, i, len(dataloader), Pixellevel_L1_Loss.item(), GAN_Loss.item(), loss_D.item()))
            print("\r[Attention Loss: %.4f] [Perceptual Loss: %.4f] Time_left: %s" %
                (Attention_Loss.item(), Perceptual_Loss.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_D)

def Continue_train_WGAN(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()

    # Initialize Generator
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    perceptualnet = utils.create_perceptualnet()

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
        discriminator = nn.DataParallel(discriminator)
        discriminator = discriminator.cuda()
        perceptualnet = nn.DataParallel(perceptualnet)
        perceptualnet = perceptualnet.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2))
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        #Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator.module, 'WGAN_%s_epoch%d_bs%d.pth' % (opt.task, epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator.module, 'WGAN_%s_iter%d_bs%d.pth' % (opt.task, iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator, 'WGAN_%s_epoch%d_bs%d.pth' % (opt.task, epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator, 'WGAN_%s_iter%d_bs%d.pth' % (opt.task, iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.RAW2RGBDataset(opt)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # For loop training
    for epoch in range(opt.epochs):
        for i, (true_input, true_target, true_sal) in enumerate(dataloader):
            
            # To device
            true_input = true_input.cuda()
            true_target = true_target.cuda()
            true_sal = true_sal.cuda()
            true_sal3 = torch.cat((true_sal, true_sal, true_sal), 1).cuda()

            # Train Discriminator
            for j in range(opt.additional_training_d):
                optimizer_D.zero_grad()

                # Generator output
                fake_target, fake_sal = generator(true_input)

                # Fake samples
                fake_block1, fake_block2, fake_block3, fake_scalar = discriminator(true_input, fake_target.detach())
                # True samples
                true_block1, true_block2, true_block3, true_scalar = discriminator(true_input, true_target)
                '''
                # Feature Matching Loss
                FM_Loss = criterion_L1(fake_block1, true_block1) + criterion_L1(fake_block2, true_block2) + criterion_L1(fake_block3, true_block3)
                '''
                # Overall Loss and optimize
                loss_D = - torch.mean(true_scalar) + torch.mean(fake_scalar)
                loss_D.backward()
     
            # Train Generator
            optimizer_G.zero_grad()
            fake_target, fake_sal = generator(true_input)

            # L1 Loss
            Pixellevel_L1_Loss = criterion_L1(fake_target, true_target)

            # Attention Loss
            true_Attn_target = true_target.mul(true_sal3)
            fake_sal3 = torch.cat((fake_sal, fake_sal, fake_sal), 1)
            fake_Attn_target = fake_target.mul(fake_sal3)
            Attention_Loss = criterion_L1(fake_Attn_target, true_Attn_target)
            
            # GAN Loss
            fake_block1, fake_block2, fake_block3, fake_scalar = discriminator(true_input, fake_target)
            GAN_Loss = - torch.mean(fake_scalar)
            
            # Perceptual Loss
            fake_target = fake_target * 0.5 + 0.5
            true_target = true_target * 0.5 + 0.5
            fake_target = utils.normalize_ImageNet_stats(fake_target)
            true_target = utils.normalize_ImageNet_stats(true_target)
            fake_percep_feature = perceptualnet(fake_target)
            true_percep_feature = perceptualnet(true_target)
            Perceptual_Loss = criterion_L1(fake_percep_feature, true_percep_feature)

            # Overall Loss and optimize
            loss = Pixellevel_L1_Loss + opt.lambda_attn * Attention_Loss + opt.lambda_gan * GAN_Loss + opt.lambda_percep * Perceptual_Loss
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Pixellevel L1 Loss: %.4f] [G Loss: %.4f] [D Loss: %.4f]" %
                ((epoch + 1), opt.epochs, i, len(dataloader), Pixellevel_L1_Loss.item(), GAN_Loss.item(), loss_D.item()))
            print("\r[Attention Loss: %.4f] [Perceptual Loss: %.4f] Time_left: %s" %
                (Attention_Loss.item(), Perceptual_Loss.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_D)
