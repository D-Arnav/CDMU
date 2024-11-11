import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR

from utils.utils import validate, vis_adv_mu_loss, AdversarialSample, create_pseudo_dl, create_noisy_dl
from utils.sfda import SFDA2
from utils.loss import CrossEntropyLabelSmooth
from utils.train import update_CV, IFA
from utils.data import get_continual_loaders

def finetune(classifier, config):
    """
    Finetunes model on a subset of target retain dl
    We modify this to SFDA finetuning as CE finetuning would cause catastrophic forgetting
    """

    target_train_dl = config['target_retain_subset_dl']
    
    sfda2 = SFDA2(classifier, target_train_dl, config, dump_name='subset')
    
    optimizer = SGD(classifier.get_parameters(), lr=config['sfda2_lr'], momentum=config['sfda2_mom'], weight_decay=config['sfda2_wt_decay'], nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: config['sfda2_lr'] * (1. + config['sfda2_lr_decay'] * float(x)) ** -config['sfda2_gamma'])
    iter_num = 0
    
    for epoch in range(config['finetune_epochs']):
        classifier.train()
        loss_val = 0
        
        for i in tqdm(range(config['iter_per_epoch']), desc=f"Epoch {epoch+1}"):
            images, indices = next(sfda2.target_train_dl)
            images = images.to(config['device'])

            sfda2.update_iter_num(iter_num)
            loss = sfda2.loss(images, indices)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            loss_val += loss.item()
            iter_num += 1
        
        # if config['save']:
        #     path = f"{config['save_path']}/{config['dataset']}/{config['backbone']}_{config['source']}_{config['target']}_{config['forget_classes']}.pt"
        #     os.makedirs(os.path.dirname(path), exist_ok=True)
        #     torch.save(classifier.state_dict(), path)
            

        loss_val /= config['iter_per_epoch']
        source_acc = validate(classifier, config['source_test_dl'].data_loader, config)
        target_acc = validate(classifier, config['target_retain_test_dl'].data_loader, config)
        forget_acc = validate(classifier, config['target_forget_dl'].data_loader, config)


        print(f"----------\n"
              f"Epoch                  : {epoch+1}\n"
              f"Train Loss             : {loss_val:.2f}\n"
              f"Source Accuracy        : {source_acc:.2f}\n" 
              f"Target Accuracy        : {target_acc:.2f}\n" 
              f"Target Forget Accuracy : {forget_acc:.2f}\n"
              f"----------\n")

    return classifier


def sample_minimax(classifier, config, vis=True):
    """
    """

    adversarial = AdversarialSample(config['forget_classes'], config)
    adversarial.learn_init(classifier)

    target_train_dl = config['target_retain_train_dl']
    
    sfda2 = SFDA2(classifier, target_train_dl, config, dump_name='retain')
    
    optimizer = SGD(classifier.get_parameters(), lr=config['sfda2_lr'], momentum=config['sfda2_mom'], weight_decay=config['sfda2_wt_decay'], nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: config['sfda2_lr'] * (1. + config['sfda2_lr_decay'] * float(x)) ** -config['sfda2_gamma'])
    mu_hist = []
    iter_num = 0
    
    for epoch in range(config['epochs']):
        classifier.train()
        classifier.zero_grad()
        loss_val = 0
        
        for i in tqdm(range(config['iter_per_epoch']), desc=f"Epoch {epoch+1}"):
            images, indices = next(sfda2.target_train_dl)
            images = images.to(config['device'])

            sfda2.update_iter_num(iter_num)
            sfda_loss = sfda2.loss(images, indices)

            sample = adversarial.sample().detach().to(config['device'])
            logits = classifier(sample)[0]
            labels = logits.detach().clone().to(config['device'])
            labels[:, config['forget_classes']] = -float('inf')
            labels = F.softmax(labels, dim=1)
            mu_loss = F.cross_entropy(logits, labels)
            
            loss = sfda_loss + mu_loss * config['minimax_alpha']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            loss_val += loss.item()
            iter_num += 1

            mu_hist.append(mu_loss.item())
            adversarial.update(classifier)
        
        # if config['save']:
        #     path = f"{config['save_path']}/{config['dataset']}/{config['backbone']}_{config['source']}_{config['target']}_{config['forget_classes']}.pt"
        #     os.makedirs(os.path.dirname(path), exist_ok=True)
        #     torch.save(classifier.state_dict(), path)
            
        if (not config['fast_train']) or (epoch == config['epochs'] - 1):
        
            loss_val /= config['iter_per_epoch']
            source_acc = validate(classifier, config['source_test_dl'].data_loader, config)
            target_acc = validate(classifier, config['target_retain_test_dl'].data_loader, config)
            forget_acc = validate(classifier, config['target_forget_dl'].data_loader, config)


            print(f"----------\n"
                f"Epoch                  : {epoch+1}\n"
                f"Train Loss             : {loss_val:.2f}\n"
                f"Adv Optim Loss         : {adversarial.hist[-1]:.2f}\n"
                f"Source Accuracy        : {source_acc:.2f}\n" 
                f"Target Accuracy        : {target_acc:.2f}\n" 
                f"Target Forget Accuracy : {forget_acc:.2f}\n"
                f"----------\n")

    if vis: vis_adv_mu_loss(adversarial.hist, mu_hist, config)

    return classifier

import os

from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from tllib.utils.data import ForeverDataIterator


def finetune_old(source_classifier, config):
    """
    
    """
    
    
    target_train_dl = config['target_retain_train_dl']
    source_test_dl = config['source_test_dl']
    target_test_dl = config['target_retain_test_dl']
    target_forget_test_dl = config['target_forget_dl']

    device = config['device']
    batch_size = config['batch']
    bottleneck = config['bottleneck']
    K = 2
    lambda_0 = 5.0
    alpha_1 = 1e-4
    alpha_2 = 1.0

    iter_per_epoch = config['iter_per_epoch']
    epochs = config['finetune_epochs']
    
    dataset = []
    for i, (images, labels) in tqdm(enumerate(target_train_dl.data_loader), desc="Creating Indices"):
        indices = torch.arange(i * batch_size, (i + 1) * batch_size)
        for b in range(images.size(0)):
            dataset.append((images[b], indices[b]))
    target_train_dl = DataLoader(dataset, batch_size, shuffle=True, num_workers=8)
    target_train_dl = ForeverDataIterator(target_train_dl)
    
    # Banks
    num_sample = len(target_train_dl.data_loader.dataset)
    feature_bank = torch.randn(num_sample, bottleneck)
    output_bank = torch.randn(num_sample, config['num_classes'])
    pseudo_bank = torch.randn(num_sample).long()

    with torch.no_grad():
        source_classifier.train()
        for i in tqdm(range(len(target_train_dl)), desc="Creating Banks"):
            images, indices = next(target_train_dl)
            images = images.to(device)
            logits, features = source_classifier(images)
            norm_features = F.normalize(features, dim=1)
            outputs = F.softmax(logits, dim=1)
            pseudo_labels = torch.argmax(outputs, dim=1)

            feature_bank[indices] = norm_features.detach().clone().cpu()
            output_bank[indices] = outputs.detach().clone().cpu()
            pseudo_bank[indices] = pseudo_labels.detach().clone().cpu()

    rho = torch.ones([config['num_classes']]).to(device) / config['num_classes']
    cov = torch.zeros(config['num_classes'], bottleneck, bottleneck).to(device)
    ave = torch.zeros(config['num_classes'], bottleneck).to(device)
    amount = torch.zeros(config['num_classes']).to(device)

    optimizer = SGD(source_classifier.get_parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-3, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: 1e-2 * (1. + 1e-3 * float(x)) ** -0.90)
    iter_num = 0
    max_iter = iter_per_epoch * epochs

    for epoch in range(epochs):
        source_classifier.train()
        loss_val = 0

        for i in tqdm(range(iter_per_epoch), desc=f"Epoch {epoch+1}"):
            images, indices = next(target_train_dl)
            images = images.to(device)
            logits, features = source_classifier(images)
            norm_features = F.normalize(features, dim=1)
            outputs = F.softmax(logits, dim=1)
            pseudo_labels = torch.argmax(outputs, dim=1)
            alpha = (1 + 10 * iter_num / max_iter) ** -5

            with torch.no_grad():
                distance = feature_bank[indices] @ feature_bank.T
                _, idx_near = torch.topk(distance, dim=-1, largest=True, k=K+1)
                idx_near = idx_near[:, 1:]
                outputs_near = output_bank[idx_near]

            ## SNC
            
            rho_batch = torch.histc(pseudo_labels, bins=config['num_classes'], min=0, max=config['num_classes'] - 1) / images.shape[0]
            rho = 0.95 * rho + 0.05 * rho_batch

            softmax_out_un = outputs.unsqueeze(1).expand(-1, K, -1).to(device)

            loss_pos = torch.mean(
                (F.kl_div(softmax_out_un, outputs_near.to(device), reduction="none").sum(dim=-1)).sum(dim=1)
            )

            mask = torch.ones((images.shape[0], images.shape[0]))
            diag_num = torch.diag(mask)
            mask_diag = torch.diag_embed(diag_num)
            mask = mask - mask_diag
            copy = outputs.T
            dot_neg = outputs @ copy
            dot_neg = ((dot_neg**2) * mask.to(device)).sum(dim=-1)
            neg_pred = torch.mean(dot_neg)
            loss_neg = neg_pred * alpha

            ## IFA

            w = source_classifier.head.weight
            ratio = lambda_0 * (iter_num / max_iter)
            amount, ave, cov = update_CV(features, pseudo_labels, amount, ave, cov, config['num_classes'], config)
            loss_ifa_, sigma2 = IFA(w, features, logits, cov, ratio, config['num_classes'], config)
            loss_ifa = alpha_1 * torch.mean(loss_ifa_)

            ## FD

            mean_score = torch.stack([torch.mean(output_bank[pseudo_bank == i], dim=0) for i in range(config['num_classes'])])
            mean_score[mean_score != mean_score] = 0.
            cov_weight = (mean_score @ mean_score.T) * (1.-torch.eye(config['num_classes']))
            cov1 = cov.view(config['num_classes'],-1).unsqueeze(1)
            cov0 = cov.view(config['num_classes'],-1).unsqueeze(0)
            cov_distance = 1 - torch.sum((cov1*cov0),dim=2) / (torch.norm(cov1, dim=2) * torch.norm(cov0, dim=2) + 1e-12)
            loss_fd = -torch.sum(cov_distance * cov_weight.to(device).detach()) / 2

            loss = loss_pos + loss_neg + (alpha_1 * loss_ifa) + (alpha_2 * loss_fd)

            loss_val += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()


       
        if (not config['fast_train']) or (epoch == config['epochs'] - 1):

            loss_val /= config['iter_per_epoch']
            source_acc = validate(source_classifier, config['source_test_dl'].data_loader, config)
            target_acc = validate(source_classifier, config['target_retain_test_dl'].data_loader, config)
            forget_acc = validate(source_classifier, config['target_forget_dl'].data_loader, config)

            print(f"----------\n"
                f"Epoch                  : {epoch+1}\n"
                f"Train Loss             : {loss_val:.2f}\n"
                f"Source Accuracy        : {source_acc:.2f}\n" 
                f"Target Accuracy        : {target_acc:.2f}\n" 
                f"Target Forget Accuracy : {forget_acc:.2f}\n"
                f"----------\n")

    
    log = f"[{config['method']}] -- [{config['source'][0]} -> {config['target'][0]}]: ({target_acc:.2f} | {forget_acc:.2f})\n"
    print(log)

    return source_classifier

def sample_minimax_old(source_classifier, config, vis=True):
    
    target_train_dl = config['target_retain_train_dl']
    source_test_dl = config['source_test_dl']
    target_test_dl = config['target_retain_test_dl']
    target_forget_test_dl = config['target_forget_dl']

    device = config['device']
    batch_size = config['batch']
    bottleneck = config['bottleneck']
    K = 2
    lambda_0 = 5.0
    alpha_1 = 1e-4
    alpha_2 = 1.0

    iter_per_epoch = config['iter_per_epoch']
    epochs = config['epochs']

    adversarial = AdversarialSample(config['forget_classes'], config)
    adversarial.learn_init(source_classifier)
    
    dataset = []
    for i, (images, labels) in tqdm(enumerate(target_train_dl.data_loader), desc="Creating Indices"):
        indices = torch.arange(i * batch_size, (i + 1) * batch_size)
        for b in range(images.size(0)):
            dataset.append((images[b], indices[b]))
    target_train_dl = DataLoader(dataset, batch_size, shuffle=True, num_workers=8)
    target_train_dl = ForeverDataIterator(target_train_dl)
    
    # Banks
    num_sample = len(target_train_dl.data_loader.dataset)
    feature_bank = torch.randn(num_sample, bottleneck)
    output_bank = torch.randn(num_sample, config['num_classes'])
    pseudo_bank = torch.randn(num_sample).long()

    with torch.no_grad():
        source_classifier.train()
        for i in tqdm(range(len(target_train_dl)), desc="Creating Banks"):
            images, indices = next(target_train_dl)
            images = images.to(device)
            logits, features = source_classifier(images)
            norm_features = F.normalize(features, dim=1)
            outputs = F.softmax(logits, dim=1)
            pseudo_labels = torch.argmax(outputs, dim=1)

            feature_bank[indices] = norm_features.detach().clone().cpu()
            output_bank[indices] = outputs.detach().clone().cpu()
            pseudo_bank[indices] = pseudo_labels.detach().clone().cpu()

    rho = torch.ones([config['num_classes']]).to(device) / config['num_classes']
    cov = torch.zeros(config['num_classes'], bottleneck, bottleneck).to(device)
    ave = torch.zeros(config['num_classes'], bottleneck).to(device)
    amount = torch.zeros(config['num_classes']).to(device)

    optimizer = SGD(source_classifier.get_parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-3, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: 1e-2 * (1. + 1e-3 * float(x)) ** -0.90)
    iter_num = 0
    max_iter = iter_per_epoch * epochs

    mu_hist = []

    for epoch in range(epochs):
        source_classifier.train()
        loss_val = 0

        for i in tqdm(range(iter_per_epoch), desc=f"Epoch {epoch+1}"):
            images, indices = next(target_train_dl)
            images = images.to(device)
            logits, features = source_classifier(images)
            norm_features = F.normalize(features, dim=1)
            outputs = F.softmax(logits, dim=1)
            pseudo_labels = torch.argmax(outputs, dim=1)
            alpha = (1 + 10 * iter_num / max_iter) ** -5

            with torch.no_grad():
                distance = feature_bank[indices] @ feature_bank.T
                _, idx_near = torch.topk(distance, dim=-1, largest=True, k=K+1)
                idx_near = idx_near[:, 1:]
                outputs_near = output_bank[idx_near]

            ## SNC
            
            rho_batch = torch.histc(pseudo_labels, bins=config['num_classes'], min=0, max=config['num_classes'] - 1) / images.shape[0]
            rho = 0.95 * rho + 0.05 * rho_batch

            softmax_out_un = outputs.unsqueeze(1).expand(-1, K, -1).to(device)

            loss_pos = torch.mean(
                (F.kl_div(softmax_out_un, outputs_near.to(device), reduction="none").sum(dim=-1)).sum(dim=1)
            )

            mask = torch.ones((images.shape[0], images.shape[0]))
            diag_num = torch.diag(mask)
            mask_diag = torch.diag_embed(diag_num)
            mask = mask - mask_diag
            copy = outputs.T
            dot_neg = outputs @ copy
            dot_neg = ((dot_neg**2) * mask.to(device)).sum(dim=-1)
            neg_pred = torch.mean(dot_neg)
            loss_neg = neg_pred * alpha

            ## IFA

            w = source_classifier.head.weight
            ratio = lambda_0 * (iter_num / max_iter)
            amount, ave, cov = update_CV(features, pseudo_labels, amount, ave, cov, config['num_classes'], config)
            loss_ifa_, sigma2 = IFA(w, features, logits, cov, ratio, config['num_classes'], config)
            loss_ifa = alpha_1 * torch.mean(loss_ifa_)

            ## FD

            mean_score = torch.stack([torch.mean(output_bank[pseudo_bank == i], dim=0) for i in range(config['num_classes'])])
            mean_score[mean_score != mean_score] = 0.
            cov_weight = (mean_score @ mean_score.T) * (1.-torch.eye(config['num_classes']))
            cov1 = cov.view(config['num_classes'],-1).unsqueeze(1)
            cov0 = cov.view(config['num_classes'],-1).unsqueeze(0)
            cov_distance = 1 - torch.sum((cov1*cov0),dim=2) / (torch.norm(cov1, dim=2) * torch.norm(cov0, dim=2) + 1e-12)
            loss_fd = -torch.sum(cov_distance * cov_weight.to(device).detach()) / 2

            sfda_loss = loss_pos + loss_neg + (alpha_1 * loss_ifa) + (alpha_2 * loss_fd)

            sample = adversarial.sample().detach().to(config['device'])
            logits = source_classifier(sample)[0]
            labels = logits.detach().clone().to(config['device'])
            labels[:, config['forget_classes']] = -float('inf')
            labels = F.softmax(labels, dim=1)
            mu_loss = F.cross_entropy(logits, labels)
    
            loss = sfda_loss + mu_loss * config['minimax_alpha']
            
            loss_val += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            mu_hist.append(mu_loss.item())
            adversarial.update(source_classifier)
            
       
        if (not config['fast_train']) or (epoch == config['epochs'] - 1):
        
            loss_val /= config['iter_per_epoch']
            source_acc = validate(source_classifier, config['source_test_dl'].data_loader, config)
            target_acc = validate(source_classifier, config['target_retain_test_dl'].data_loader, config)
            forget_acc = validate(source_classifier, config['target_forget_dl'].data_loader, config)


            print(f"----------\n"
                f"Epoch                  : {epoch+1}\n"
                f"Train Loss             : {loss_val:.2f}\n"
                f"Adv Optim Loss         : {adversarial.hist[-1]:.2f}\n"
                f"Source Accuracy        : {source_acc:.2f}\n" 
                f"Target Accuracy        : {target_acc:.2f}\n" 
                f"Target Forget Accuracy : {forget_acc:.2f}\n"
                f"----------\n")

    if vis: vis_adv_mu_loss(adversarial.hist, mu_hist, config)

    
    log = f"[{config['method']}] -- [{config['source'][0]} -> {config['target'][0]}]: ({target_acc:.2f} | {forget_acc:.2f})\n"
    print(log)
    # with open('log.txt', 'a') as f:
        # f.write(log)

    return source_classifier


def sample_minimax_continual(target_classifier, forget_classes, forget_classes_all, config):
    """
    Task 2: Retain: {4-65}
            Forget: {2,3}
    
    Task 3: Retain {6-65}
            Forget: {4,5}
    """

    get_continual_loaders(forget_classes_all, config)

    target_train_dl = config['continual_target_retain_subset_dl']
    source_test_dl = config['continual_source_test_dl']
    target_retain_test_dl = config['continual_target_retain_test_dl']
    target_forget_dl = config['continual_target_forget_dl']

    device = config['device']
    batch_size = config['batch']
    bottleneck = config['bottleneck']
    K = 2
    lambda_0 = 5.0
    alpha_1 = 1e-4
    alpha_2 = 1.0

    iter_per_epoch = config['iter_per_epoch']
    epochs = config['finetune_epochs']

    adversarial = AdversarialSample(forget_classes, config)
    adversarial.learn_init(target_classifier)
    
    dataset = []
    for i, (images, labels) in tqdm(enumerate(target_train_dl.data_loader), desc="Creating Indices"):
        indices = torch.arange(i * batch_size, (i + 1) * batch_size)
        for b in range(images.size(0)):
            dataset.append((images[b], indices[b]))
    target_train_dl = DataLoader(dataset, batch_size, shuffle=True, num_workers=8)
    target_train_dl = ForeverDataIterator(target_train_dl)
    
    # Banks
    num_sample = len(target_train_dl.data_loader.dataset)
    feature_bank = torch.randn(num_sample, bottleneck)
    output_bank = torch.randn(num_sample, config['num_classes'])
    pseudo_bank = torch.randn(num_sample).long()

    with torch.no_grad():
        target_classifier.train()
        for i in tqdm(range(len(target_train_dl)), desc="Creating Banks"):
            images, indices = next(target_train_dl)
            images = images.to(device)
            logits, features = target_classifier(images)
            norm_features = F.normalize(features, dim=1)
            outputs = F.softmax(logits, dim=1)
            pseudo_labels = torch.argmax(outputs, dim=1)

            feature_bank[indices] = norm_features.detach().clone().cpu()
            output_bank[indices] = outputs.detach().clone().cpu()
            pseudo_bank[indices] = pseudo_labels.detach().clone().cpu()

    rho = torch.ones([config['num_classes']]).to(device) / config['num_classes']
    cov = torch.zeros(config['num_classes'], bottleneck, bottleneck).to(device)
    ave = torch.zeros(config['num_classes'], bottleneck).to(device)
    amount = torch.zeros(config['num_classes']).to(device)

    optimizer = SGD(target_classifier.get_parameters(), lr=3e-2, momentum=0.9, weight_decay=1e-3, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: 3e-2 * (1. + 1e-3 * float(x)) ** -0.90)
    iter_num = 0
    max_iter = iter_per_epoch * epochs

    mu_hist = []

    for epoch in range(epochs):
        target_classifier.train()
        loss_val = 0

        for i in tqdm(range(iter_per_epoch), desc=f"Epoch {epoch+1}"):
            images, indices = next(target_train_dl)
            images = images.to(device)
            logits, features = target_classifier(images)
            norm_features = F.normalize(features, dim=1)
            outputs = F.softmax(logits, dim=1)
            pseudo_labels = torch.argmax(outputs, dim=1)
            alpha = (1 + 10 * iter_num / max_iter) ** -5

            with torch.no_grad():
                distance = feature_bank[indices] @ feature_bank.T
                _, idx_near = torch.topk(distance, dim=-1, largest=True, k=K+1)
                idx_near = idx_near[:, 1:]
                outputs_near = output_bank[idx_near]

            ## SNC
            
            rho_batch = torch.histc(pseudo_labels, bins=config['num_classes'], min=0, max=config['num_classes'] - 1) / images.shape[0]
            rho = 0.95 * rho + 0.05 * rho_batch

            softmax_out_un = outputs.unsqueeze(1).expand(-1, K, -1).to(device)

            loss_pos = torch.mean(
                (F.kl_div(softmax_out_un, outputs_near.to(device), reduction="none").sum(dim=-1)).sum(dim=1)
            )

            mask = torch.ones((images.shape[0], images.shape[0]))
            diag_num = torch.diag(mask)
            mask_diag = torch.diag_embed(diag_num)
            mask = mask - mask_diag
            copy = outputs.T
            dot_neg = outputs @ copy
            dot_neg = ((dot_neg**2) * mask.to(device)).sum(dim=-1)
            neg_pred = torch.mean(dot_neg)
            loss_neg = neg_pred * alpha

            ## IFA

            w = target_classifier.head.weight
            ratio = lambda_0 * (iter_num / max_iter)
            amount, ave, cov = update_CV(features, pseudo_labels, amount, ave, cov, config['num_classes'], config)
            loss_ifa_, sigma2 = IFA(w, features, logits, cov, ratio, config['num_classes'], config)
            loss_ifa = alpha_1 * torch.mean(loss_ifa_)

            ## FD

            mean_score = torch.stack([torch.mean(output_bank[pseudo_bank == i], dim=0) for i in range(config['num_classes'])])
            mean_score[mean_score != mean_score] = 0.
            cov_weight = (mean_score @ mean_score.T) * (1.-torch.eye(config['num_classes']))
            cov1 = cov.view(config['num_classes'],-1).unsqueeze(1)
            cov0 = cov.view(config['num_classes'],-1).unsqueeze(0)
            cov_distance = 1 - torch.sum((cov1*cov0),dim=2) / (torch.norm(cov1, dim=2) * torch.norm(cov0, dim=2) + 1e-12)
            loss_fd = -torch.sum(cov_distance * cov_weight.to(device).detach()) / 2

            sfda_loss = loss_pos + loss_neg + (alpha_1 * loss_ifa) + (alpha_2 * loss_fd)

            sample = adversarial.sample().detach().to(config['device'])
            logits = target_classifier(sample)[0]
            labels = logits.detach().clone().to(config['device'])
            labels[:, forget_classes_all] = -float('inf')
            labels = F.softmax(labels, dim=1)
            mu_loss = F.cross_entropy(logits, labels)
    
            loss = sfda_loss + mu_loss * config['minimax_alpha']
            
            loss_val += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            mu_hist.append(mu_loss.item())
            adversarial.update(target_classifier)
            
       
        if (not config['fast_train']) or (epoch == config['epochs'] - 1):
        
            loss_val /= config['iter_per_epoch']
            source_acc = validate(target_classifier, source_test_dl.data_loader, config)
            target_acc = validate(target_classifier, target_retain_test_dl.data_loader, config)
            forget_acc = validate(target_classifier, target_forget_dl.data_loader, config)

            print(f"----------\n"
                f"Epoch                  : {epoch+1}\n"
                f"Train Loss             : {loss_val:.2f}\n"
                f"Adv Optim Loss         : {adversarial.hist[-1]:.2f}\n"
                f"Source Accuracy        : {source_acc:.2f}\n" 
                f"Target Accuracy        : {target_acc:.2f}\n" 
                f"Target Forget Accuracy : {forget_acc:.2f}\n"
                f"----------\n")

    
    log = f"[{config['method']}] -- [{config['source'][0]} -> {config['target'][0]}]: ({target_acc:.2f} | {forget_acc:.2f})\n"
    print(log)
    # with open('log.txt', 'a') as f:
        # f.write(log)

    return target_classifier


def unsir(source_classifier, config):
    """
    UNSIR Unlearning Method
    """

    pseudo_target_dl = create_pseudo_dl(source_classifier, config['target_retain_subset_dl'].data_loader, config)
    noisy_dl = create_noisy_dl(source_classifier, pseudo_target_dl, config, num_reg=256)

    optimizer = SGD(source_classifier.get_parameters(),
                    lr=0.1, momentum=0.9, weight_decay=1e-3, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: 1e-1 * (1. + 1e-3 * float(x)) ** -0.75)
    
    for epoch in range(1):
        source_classifier.train()
        for i, (images, labels) in enumerate(noisy_dl):
            images, labels = images.to(config['device']), labels.to(config['device'])
            logits = source_classifier(images)[0]
            
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            if i == 8:
                break

    optimizer = SGD(source_classifier.get_parameters(),
                    lr=0.1, momentum=0.9, weight_decay=1e-3, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: 1e-1 * (1. + 1e-3 * float(x)) ** -0.75)

    for epoch in range(1):
        source_classifier.train()
        for i, (images, labels) in enumerate(pseudo_target_dl):
            images, labels = images.to(config['device']), labels.to(config['device'])
            logits = source_classifier(images)[0]
            
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

    source_acc = validate(source_classifier, config['source_test_dl'].data_loader, config)
    target_acc = validate(source_classifier, config['target_retain_test_dl'].data_loader, config)
    forget_acc = validate(source_classifier, config['target_forget_dl'].data_loader, config)

    print(f"----------\n"
        f"Epoch                  : {epoch+1}\n"
        f"Source Accuracy        : {source_acc:.2f}\n" 
        f"Target Accuracy        : {target_acc:.2f}\n" 
        f"Target Forget Accuracy : {forget_acc:.2f}\n"
        f"----------\n")

    return source_classifier