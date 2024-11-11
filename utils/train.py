import os

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from utils.utils import num_correct, validate
from utils.sfda import SFDA2
from utils.loss import CrossEntropyLabelSmooth


def train_source(classifier, config, retain=False, save=True, smooth=True):
    """
    Trains a model with source dataloader 
    """

    if retain:
        train_dl = config['source_retain_train_dl']
        test_dl = config['source_retain_test_dl']
    else:
        train_dl = config['source_train_dl']
        test_dl = config['source_test_dl']

    if smooth:
        loss_fn = CrossEntropyLabelSmooth(config)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = SGD(classifier.get_parameters(),
                    lr=1e-2, momentum=0.9, weight_decay=1e-3, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: 1e-2 * (1. + 1e-3 * float(x)) ** -0.75)

    for epoch in range(config['source_epochs']):
        classifier.train()
        loss_val = 0
        
        for i in tqdm(range(config['iter_per_epoch']), desc=f"Epoch {epoch+1}"):
            images, labels = next(train_dl)
            images, labels = images.to(config['device']), labels.to(config['device'])
            logits = classifier(images)[0]
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            loss_val += loss.item()

        loss_val /= config['iter_per_epoch']

        if (not config['fast_train']) or (epoch == config['source_epochs'] - 1):

            if save:
                path = f"{config['save_path']}/{config['dataset']}/{config['backbone']}_{config['source']}.pt"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(classifier.state_dict(), path)
        
            acc = validate(classifier, config['source_test_dl'].data_loader, config)        

            print(f"----------\n"
                    f"Loss     : {loss_val:.2f}\n"
                    f"Accuracy : {acc:.2f}\n" 
                    f"----------\n")

    return classifier


def train_sfda2(classifier, config, save=False):
    """
    Trains domain adaptation model using SFDA^2 Algorithm
    """
    
    target_train_dl = config['target_retain_train_dl']
    
    sfda2 = SFDA2(classifier, target_train_dl, config, dump_name='retain')
    
    optimizer = SGD(classifier.get_parameters(), lr=config['sfda2_lr'], momentum=config['sfda2_mom'], weight_decay=config['sfda2_wt_decay'], nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: config['sfda2_lr'] * (1. + config['sfda2_lr_decay'] * float(x)) ** -config['sfda2_gamma'])
    iter_num = 0
    
    for epoch in range(config['epochs']):
        classifier.train()
        loss_val = 0
        
        for i in tqdm(range(config['iter_per_epoch']), desc=f"Epoch {epoch+1}"):
            classifier.train()
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

        if (not config['fast_train']) or (epoch == config['epochs'] - 1):
        
            if save:
                path = f"{config['save_path']}/{config['dataset']}/{config['backbone']}_{config['source']}_{config['target']}_{config['forget_classes']}_{save}.pt"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(classifier.state_dict(), path)

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


import os

from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from tllib.utils.data import ForeverDataIterator

def train_sfda2_old(source_classifier, config, save=False):
    
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

        # if not config.dont_save:
        #     torch.save(source_classifier.state_dict(), f"{config.save_path}/{config.domain}/{config.backbone}_{config.source}_{config.target}_{config.forget_classes}.pt")
            
       
        if (not config['fast_train']) or (epoch == config['epochs'] - 1):
        
            if save:
                path = f"{config['save_path']}/{config['dataset']}/{config['backbone']}_{config['source']}_{config['target']}_{config['forget_classes']}_{save}.pt"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(source_classifier.state_dict(), path)

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
    # with open('log.txt', 'a') as f:
    #     f.write(log)

    return source_classifier


def update_CV(features, labels, Amount, Ave, Cov, C, config):
    N = features.size(0)
    A = features.size(1)

    NxCxFeatures = features.view(N, 1, A).expand(N, C, A)
    onehot = torch.zeros(N, C).to(config['device'])
    onehot.scatter_(1, labels.view(-1, 1), 1)

    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A) # mask

    features_by_sort = NxCxFeatures.mul(NxCxA_onehot) # masking

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1

    ave_CxA = features_by_sort.sum(0) / Amount_CxA

    var_temp = features_by_sort - ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

    var_temp = torch.bmm(
        var_temp.permute(1, 2, 0),
        var_temp.permute(1, 0, 2)
    ).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

    sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

    sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

    weight_CV = sum_weight_CV.div(
        sum_weight_CV + Amount.view(C, 1, 1).expand(C, A, A)
    )
    weight_CV[weight_CV != weight_CV] = 0

    weight_AV = sum_weight_AV.div(
        sum_weight_AV + Amount.view(C, 1).expand(C, A)
    )
    weight_AV[weight_AV != weight_AV] = 0

    additional_CV = weight_CV.mul(1 - weight_CV).mul(
        torch.bmm(
            (Ave - ave_CxA).view(C, A, 1),
            (Ave - ave_CxA).view(C, 1, A)
        )
    )

    Cov = (Cov.mul(1 - weight_CV).detach() + var_temp.mul(weight_CV)) + additional_CV
    Ave = (Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()
    Amount = Amount + onehot.sum(0)
    return Amount, Ave, Cov

def IFA(w, features, logit, cv_matrix, ratio, C, config):
    N = features.size(0)
    A = features.size(1)
    log_prob_ifa_ = []
    sigma2_ = []
    pseudo_labels = torch.argmax(logit, dim=1).detach()
    for i in range(C):
        labels = (torch.ones(N)*i).to(config['device']).long()
        NxW_ij = w.expand(N, C, A)
        NxW_kj = torch.gather(NxW_ij, 1, labels.view(N, 1, 1).expand(N, C, A))
        CV_temp = cv_matrix[pseudo_labels]

        sigma2 = ratio * torch.bmm(torch.bmm(NxW_ij-NxW_kj, CV_temp), (NxW_ij - NxW_kj).permute(0, 2, 1))
        with torch.no_grad():
            sigma2_.append(torch.mean(sigma2))
        sigma2 = sigma2.mul(torch.eye(C).to(config['device']).expand(N, C, C)).sum(2).view(N, C)
        ifa_logit = logit + 0.5 * sigma2
        log_prob_ifa_.append(F.cross_entropy(ifa_logit, labels, reduction='none'))
    log_prob_ifa = torch.stack(log_prob_ifa_)
    loss = torch.sum(2 * log_prob_ifa.T, dim=1)
    return loss, torch.stack(sigma2_)