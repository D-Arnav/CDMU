import os

from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from tllib.utils.data import ForeverDataIterator

from utils.utils import load, dump


class SFDA2:
    """
    Implementation of the SFDA2++ Source-free Domain Adaptation Method
    FD Not implemented as it doesn't function target label space is a subset of source label space
    """
    
    
    def __init__(self, classifier, target_train_dl, config, dump_name=None):
        self.classifier = classifier
        self.target_train_dl = target_train_dl
        self.config = config
        self.rho = torch.ones([config['num_classes']]).to(config['device']) / config['num_classes']
        self.cov = torch.zeros(config['num_classes'], config['bottleneck'], config['bottleneck']).to(config['device'])
        self.ave = torch.zeros(config['num_classes'], config['bottleneck']).to(config['device'])
        self.amount = torch.zeros(config['num_classes']).to(config['device'])
        self.iter_num = 0
        self.max_iter = config['iter_per_epoch'] * config['epochs']
        self.dump = dump_name
        
        self._convert_index_dl()
        self._set_hyperparams()
        self._create_banks()

    def _set_hyperparams(self):
        self.config['sfda2_K'] = 5
        self.config['sfda2_lambda_0'] = 5.0
        self.config['sfda2_alpha_1'] = 1e-4
        self.config['sfda2_alpha_2'] = 10.0
        self.config['sfda2_lr'] = 1e-2
        self.config['sfda2_mom'] = 0.9
        self.config['sfda2_lr_decay'] = 1e-3
        self.config['sfda2_wt_decay']  = 1e-3
        self.config['sfda2_gamma'] = 0.9

    def _create_banks(self):
        
        if self.dump is not None:
            path = os.path.join(self.config['dump_path'], self.config['dataset'], self.config['source'], f"{self.config['target']}_sfda_banks_{self.dump}.p")
            if os.path.exists(path):
                self.banks = load(path)
                return

        N = len(self.target_train_dl.data_loader.dataset)
        self.banks = {
            'feature': torch.randn(N, self.config['bottleneck']),
            'output': torch.randn(N, self.config['num_classes']),
            'pseudo': torch.randn(N).long()
        }

        with torch.no_grad():
            self.classifier.train()
            for i in tqdm(range(len(self.target_train_dl.data_loader)), desc="Creating SFDA2 Banks"):
                images, indices = next(self.target_train_dl)
                images = images.to(self.config['device'])
                logits, features = self.classifier(images)
                norm_features = F.normalize(features, dim=1)
                outputs = F.softmax(logits, dim=1)
                pseudo_labels = torch.argmax(outputs, dim=1)

                self.banks['feature'][indices] = norm_features.detach().clone().cpu()
                self.banks['output'][indices] = outputs.detach().clone().cpu()
                self.banks['pseudo'][indices] = pseudo_labels.detach().clone().cpu()

        if self.dump is not None:
            dump(self.banks, path)
    
    def _IFA(self, w, features, logit, ratio):
        N = features.size(0)
        C = self.config['num_classes']
        A = features.size(1)
        log_prob_ifa_ = []
        sigma2_ = []
        pseudo_labels = torch.argmax(logit, dim=1).detach()
        for i in range(C):
            labels = (torch.ones(N)*i).to(self.config['device']).long()
            NxW_ij = w.expand(N, C, A)
            NxW_kj = torch.gather(NxW_ij, 1, labels.view(N, 1, 1).expand(N, C, A))
            CV_temp = self.cov[pseudo_labels]

            sigma2 = ratio * torch.bmm(torch.bmm(NxW_ij-NxW_kj, CV_temp), (NxW_ij - NxW_kj).permute(0, 2, 1))
            with torch.no_grad():
                sigma2_.append(torch.mean(sigma2))
            sigma2 = sigma2.mul(torch.eye(C).to(self.config['device']).expand(N, C, C)).sum(2).view(N, C)
            ifa_logit = logit + 0.5 * sigma2
            log_prob_ifa_.append(F.cross_entropy(ifa_logit, labels, reduction='none'))
        log_prob_ifa = torch.stack(log_prob_ifa_)
        loss = torch.sum(2 * log_prob_ifa.T, dim=1)
        return loss, torch.stack(sigma2_)
    
    def _update_CV(self, features, labels):

        N = features.size(0)
        C = self.config['num_classes']
        A = features.size(1)

        NxCxFeatures = features.view(N, 1, A).expand(N, C, A)
        onehot = torch.zeros(N, C).to(self.config['device'])
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A) 

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot) 

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
            sum_weight_CV + self.amount.view(C, 1, 1).expand(C, A, A)
        )
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.amount.view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.ave - ave_CxA).view(C, A, 1),
                (self.ave - ave_CxA).view(C, 1, A)
            )
        )

        self.cov = (self.cov.mul(1 - weight_CV).detach() + var_temp.mul(weight_CV)) + additional_CV
        self.ave = (self.ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()
        self.amount = self.amount + onehot.sum(0)

    def _convert_index_dl(self):
        dataset = []
        for i, (images, labels) in tqdm(enumerate(self.target_train_dl.data_loader), desc="Creating SFDA2 Indices"):
            indices = torch.arange(i * self.config['batch'], (i + 1) * self.config['batch'])
            for b in range(images.size(0)):
                dataset.append((images[b], indices[b]))
        dl = DataLoader(dataset, self.config['batch'], shuffle=True, num_workers=self.config['workers'])
        self.target_train_dl = ForeverDataIterator(dl)

    def update_iter_num(self, iter_num):
        self.iter_num = iter_num

    def loss(self, images, indices):
        self.classifier.train()
        logits, features = self.classifier(images)
        outputs = F.softmax(logits, dim=1)
        pseudo_labels = torch.argmax(outputs, dim=1)
        alpha = (1 + 10 * self.iter_num / self.max_iter) ** -5

        with torch.no_grad():
            distance = self.banks['feature'][indices] @ self.banks['feature'].T
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=self.config['sfda2_K']+1)
            idx_near = idx_near[:, 1:]
            outputs_near = self.banks['output'][idx_near]

        # SNC

        rho_batch = torch.histc(pseudo_labels, bins=self.config['num_classes'], min=0, max=self.config['num_classes'] - 1) / images.shape[0]
        self.rho = 0.95 * self.rho + 0.05 * rho_batch

        softmax_out_un = outputs.unsqueeze(1).expand(-1, self.config['sfda2_K'], -1).to(self.config['device'])

        loss_pos = torch.mean(
            (F.kl_div(softmax_out_un, outputs_near.to(self.config['device']), reduction="none").sum(dim=-1)).sum(dim=1)
        )

        mask = torch.ones((images.shape[0], images.shape[0]))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag
        copy = outputs.T
        dot_neg = outputs @ copy
        dot_neg = ((dot_neg**2) * mask.to(self.config['device'])).sum(dim=-1)
        neg_pred = torch.mean(dot_neg)
        loss_neg = neg_pred * alpha

        # IFA

        w = self.classifier.head.weight
        ratio = self.config['sfda2_lambda_0'] * (self.iter_num / self.max_iter)
        self._update_CV(features, pseudo_labels)
        loss_ifa_, sigma2 = self._IFA(w, features, logits, ratio)
        loss_ifa = self.config['sfda2_alpha_1'] * torch.mean(loss_ifa_)

        mean_score = torch.stack([torch.mean(self.banks['output'][self.banks['pseudo']==i], dim=0) for i in range(self.config['num_classes'])])   
        mean_score[mean_score != mean_score] = 0

        cov_weight = (mean_score @ mean_score.T) * (1.-torch.eye(self.config['num_classes']))

        Cov1 = self.cov.view(self.config['num_classes'],-1).unsqueeze(1)
        Cov0 = self.cov.view(self.config['num_classes'],-1).unsqueeze(0)

        cov_dist_num = torch.sum((Cov1*Cov0),dim=2)
        cov_dist_den = (torch.norm(Cov1, dim=2) * torch.norm(Cov0, dim=2) + 1e-12)
        cov_distance = 1 - cov_dist_num / cov_dist_den
        loss_fd = -torch.sum(cov_distance * cov_weight.to(self.config['device']).detach()) / 2

        return loss_pos + loss_neg + self.config['sfda2_alpha_1'] * loss_ifa  +  self.config['sfda2_alpha_2'] * loss_fd
    