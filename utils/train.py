import os

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from utils.utils import num_correct, validate, dump, load, AdversarialSample, Classifier_Mask
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

    path = os.path.join(config['dump_path'], config['dataset'], config['source'], config['target'], 'banks.p')

    if not os.path.exists(path):
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

        banks = (feature_bank, output_bank, pseudo_bank)
        dump(banks, path)

    (feature_bank, output_bank, pseudo_bank) = load(path)

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
            
            rho_batch = torch.histc(pseudo_labels.float(), bins=config['num_classes'], min=0, max=config['num_classes'] - 1) / images.shape[0]
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
            mean_score = mean_score.detach().cpu()
            with torch.autograd.graph.save_on_cpu():
                cov_weight = (mean_score @ mean_score.T) * (1.-torch.eye(config['num_classes']))
                cov1 = cov.view(config['num_classes'],-1).unsqueeze(1).half()
                cov0 = cov.view(config['num_classes'],-1).unsqueeze(0).half()
                cov_distance = 1 - torch.sum((cov1*cov0),dim=2) / (torch.norm(cov1, dim=2) * torch.norm(cov0, dim=2) + 1e-12).float()
                loss_fd = -torch.sum(cov_distance * cov_weight.to(device).detach()) / 2

            loss = loss_pos + loss_neg + (alpha_1 * loss_ifa) + (alpha_2 * loss_fd)

            loss_val += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            torch.cuda.empty_cache()
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


def train_g_sfda(source_classifier, config):
    """

    """

    source_classifier = Classifier_Mask(source_classifier)

    optimizer = SGD(source_classifier.get_parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-3, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: 1e-2 * (1. + 1e-3 * float(x)) ** -0.90)

    dataset = []
    for i, (images, labels) in tqdm(enumerate(config['target_retain_train_dl'].data_loader), desc="Creating Indices"):
        indices = torch.arange(i * config['batch'], (i + 1) * config['batch'])
        for b in range(images.size(0)):
            dataset.append((images[b], indices[b]))
    target_train_dl = DataLoader(dataset, config['batch'], shuffle=True, num_workers=8)
    target_train_dl = ForeverDataIterator(target_train_dl)

    num_sample = len(target_train_dl.data_loader.dataset)
    feature_bank = torch.randn(num_sample, config['bottleneck'])
    output_bank = torch.randn(num_sample, config['num_classes'])
    pseudo_bank = torch.randn(num_sample).long()

    with torch.no_grad():
        source_classifier.train()
        for i in tqdm(range(len(target_train_dl)), desc="Creating Banks"):
            images, indices = next(target_train_dl)
            images = images.to(config['device'])
            logits, features = source_classifier(images)
            norm_features = F.normalize(features, dim=1)
            outputs = F.softmax(logits, dim=1)

            feature_bank[indices] = norm_features.detach().clone().cpu()
            output_bank[indices] = outputs.detach().clone().cpu()

    epochs = config['epochs']
    iter_per_epoch = config['iter_per_epoch']

    iter_num = 0
    max_iter = iter_per_epoch * epochs

    for epoch in range(epochs):
        source_classifier.train()
        loss_val = 0

        for i in tqdm(range(iter_per_epoch), desc=f"Epoch {epoch+1}"):
            images, indices = next(target_train_dl)
            images = images.to(config['device'])

            logits, features, masks = source_classifier(images)
            output = F.softmax(logits, dim=1)
            output_re = output.unsqueeze(1)

            with torch.no_grad():
                feature_bank[indices].fill_(-0.1)
                output_f = F.normalize(features).cpu().detach().clone()

                distance = output_f @ feature_bank.t()
                _, idx_near = torch.topk(distance, dim=-1, largest=True, k=2)
                score_near = output_bank[idx_near]  
                score_near = score_near.permute(0, 2, 1)

                feature_bank[indices] = output_f.detach().clone().cpu()
                output_bank[indices] = output.detach().clone().cpu()


            const = torch.log(torch.bmm(output_re, score_near.to(config['device']))).sum(-1)
            loss_const = -torch.mean(const)

            msoftmax = outputs.mean(dim=0)
            im_div = torch.sum(msoftmax * torch.log(msoftmax + 1e5))
            loss = im_div + loss_const 

            loss_val += loss.item()

            optimizer.zero_grad()
            loss.backward()

            for n, p in classifier.named_parameters():
                if n.find('bias') == -1:
                    mask_ = ((1 - masks_old)).view(-1, 1).expand(256, config['bottleneck']).cuda()
                p.grad.data *= mask_
            else:  #no bias here
                mask_ = ((1 - masks_old)).squeeze().cuda()
                p.grad.data *= mask_

            for n, p in classifier.classifier.head.named_parameters():
                if args.layer == 'wn' and n.find('weight_v') != -1:
                    masks__ = masks_old.view(1, -1).expand(args.class_num, 256)
                    mask_ = ((1 - masks__)).cuda()
                    #print(n,p.grad.shape)
                    p.grad.data *= mask_
                if args.layer == 'linear':
                    masks__ = masks_old.view(1, -1).expand(args.class_num, 256)
                    mask_ = ((1 - masks__)).cuda()
                    #print(n,p.grad.shape)
                    p.grad.data *= mask_

            for n, p in classifier.classifier.bottleneck.named_parameters():
                mask_ = ((1 - masks_old)).view(-1).cuda()
                p.grad.data *= mask_

            torch.nn.utils.clip_grad_norm(classifier.classifier.bottleneck.parameters(), 10000)

            
            optimizer.step()
            lr_scheduler.step()

        
        if (not config['fast_train']) or (epoch == config['epochs'] - 1):
        
            # if save:
            #     path = f"{config['save_path']}/{config['dataset']}/{config['backbone']}_{config['source']}_{config['target']}_{config['forget_classes']}_{save}.pt"
            #     os.makedirs(os.path.dirname(path), exist_ok=True)
            #     torch.save(classifier.state_dict(), path)

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

def sfda_apa_u(source_classifier, config):
    """
    APAuSF Domain Adaptaion Method
    2024-11-13_17.03.01:INFO:Namespace(model='APAuSF', 
    base_model='Base', base_net='ResNet50', 
    source_checkpoint='checkpoint_base/Base_Office-Home_Real_World_2022-00-00_00.00.00_0_5000.pth', 
    restore_checkpoint=None, restore_iter=0, dataset='Office-Home', datasets_dir='./dataset', 
    source_path='data/Real_World.txt', target_path='data/Art.txt', test_path=None, num_workers=2, 
    rand_aug_n=3, rand_aug_m=2.0, source_rand_aug_size=0, target_rand_aug_size=0, config='config/dann.yml', 
    save_interval=10000, eval_interval=1000, train_source_steps=10000, save_source_interval=10000, 
    train_steps=10000, iter=-1, batch_size=16, class_num=65, eval_source=False, eval_target=True, 
    eval_test=True, save_checkpoint=True, save_optimizer=False, lr=0.001, lr_momentum=0.9, lr_wd=0.0005, l
    r_scheduler_gamma=0.0001, lr_scheduler_decay_rate=0.75, lr_scheduler_rate=1, gpu_id='0', random_seed=0, 
    timestamp='2024-11-13_17.03.01', use_file_logger=True, log_dir='log/APA/', train_source_sampler=None, 
    train_target_sampler='ClassBalancedBatchSampler', n_way=31, k_shot=1, yhat_update_freq=100, 
    confidence_threshold=None, balance_domain=True, center_crop=False, bottleneck_dim=2048, 
    use_bottleneck=True, use_bottleneck_dropout=False, use_dropout=False, use_hidden_layer=False, 
    l2_normalize=True, temperature=0.05, class_criterion='CrossEntropyLoss', self_training_loss_weight=1.0, 
    self_training_conf=0.75, VAT_xi=10.0, VAT_eps=30.0, VAT_iter=1, vat_loss_weight=0.1, fixmatch_loss_threshold=0.97, 
    fixmatch_loss_weight=0.0, save_eval_result=False)
    """


    vat_iter = 1
    vat_xi = 10
    vat_eps = 30
    vat_alpha = 0.1
    st_conf = 0.75

    optimizer = SGD(source_classifier.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-3, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: 1e-2 * (1. + 1e-3 * float(x)) ** -0.9)

    adversarial = AdversarialSample(config['forget_classes'], config)
    adversarial.learn_init(source_classifier)
    
    for epoch in range(config['epochs']):
        source_classifier.train()
        loss_val = 0
        
        for i in tqdm(range(config['iter_per_epoch']), desc=f"Epoch {epoch+1}"):
            source_classifier.train()
            images, indices = next(config['target_retain_train_dl'])
            images = images.to(config['device'])
            
            logits, features = source_classifier(images)
            
            # ST

            confs, pseudo_labels = F.softmax(logits, dim=1).max(dim=1)

            st_loss = (F.cross_entropy(logits, pseudo_labels, reduction='none') * (confs >= st_conf).float().detach()).mean()

            # VAT

            d = torch.rand(features.shape).sub(0.5).to(config['device'])
            outputs = F.softmax(logits, dim=1).detach().clone()

            for i in range(vat_iter):
                d = vat_xi * l2_normalize(d)
                d.requires_grad_()
                outputs_ = source_classifier.head(features.detach() + d)
                logp_hat = F.log_softmax(outputs_, dim=1)
                adv_dist = F.kl_div(logp_hat, outputs, reduction='batchmean')
                adv_dist.backward()
                d = l2_normalize(d.grad)
                source_classifier.zero_grad()

            r_adv = d * vat_eps
            act = features * r_adv.detach().clone()

            outputs_ = source_classifier.head(act)
            logp_hat = F.log_softmax(outputs_, dim=1)
            adap_loss = F.kl_div(logp_hat, outputs, reduction='batchmean')

            loss = st_loss + adap_loss * vat_alpha
            
            sample = adversarial.sample().detach().to(config['device'])
            logits = source_classifier(sample)[0]
            labels = logits.detach().clone().to(config['device'])
            labels[:, config['forget_classes']] = -float('inf')
            labels = F.softmax(labels, dim=1)
            outputs = F.softmax(logits, dim=1)
            mu_loss = F.cross_entropy(outputs, labels)
            
            loss += mu_loss * 0.01

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            loss_val += loss.item()
            adversarial.update(source_classifier)

        if (not config['fast_train']) or (epoch == config['epochs'] - 1):
        
            # if save:
            #     path = f"{config['save_path']}/{config['dataset']}/{config['backbone']}_{config['source']}_{config['target']}_{config['forget_classes']}_{save}.pt"
            #     os.makedirs(os.path.dirname(path), exist_ok=True)
            #     torch.save(classifier.state_dict(), path)

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

    return classifier


def l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d