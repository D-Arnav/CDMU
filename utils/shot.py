from copy import deepcopy

import numpy as np

import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F_
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from tllib.utils.data import ForeverDataIterator

from utils.utils import num_correct, validate
from utils.loss import CrossEntropyLabelSmooth
from utils.forget import AdversarialSample
    
def train_shot(classifier, config):

    """
    Trains SHOT++ Model
    python image_target.py --gpu_id 0 --seed 2021 --da pda --dset office-home --s 0 --output_src ckps/source/ --output ckps/target/ --cls_par 0.3 --ssl 0.6
    """

    adversarial = AdversarialSample(config['forget_classes'], config, smooth=False)
    adversarial.learn_init(classifier)
    
    path = f"{config['save_path']}/{config['dataset']}/{config['backbone']}_{config['source']}_{config['target']}_shot_rotate.pt"
    if os.path.exists(path):
        print('')
        R = FeatClassifier(num_classes=4, bottleneck=2*config['bottleneck']).to(config['device'])
        R.load_state_dict(torch.load(path, map_location=config['device'])) 
    else:
        R = train_target_rot(deepcopy(classifier), config)
    
    B = deepcopy(classifier.backbone)
    F = deepcopy(classifier.bottleneck)
    H = deepcopy(classifier.head)

    for k, v in H.named_parameters():
        v.requires_grad = False
    
    param_group = []
    for k, v in B.named_parameters():
        v.requires_grad = True
        param_group += [{'params': v, 'lr': 1e-1 * 1.0}]

    for k, v in F.named_parameters():
        v.requires_grad = True
        param_group += [{'params': v, 'lr': 1e-1 * 1.0}]

    for k, v in R.named_parameters():
        v.requires_grad = True
        param_group += [{'params': v, 'lr': 1e-1 * 1.0}]
    
    optimizer = SGD(param_group)
    lr_scheduler = LambdaLR(optimizer, lambda x: 1e-1 * (1. + 1e-3 * float(x)) ** -0.9)

    dataset = []
    for i, (images, labels) in tqdm(enumerate(config['target_retain_train_dl'].data_loader), desc="Creating SHOT Indices"):
        indices = torch.arange(i * config['batch'], (i + 1) * config['batch'])
        for b in range(images.size(0)):
            dataset.append((images[b], indices[b]))
    dl = DataLoader(dataset, config['batch'], shuffle=True, num_workers=config['workers'], drop_last=True)
    
    target_train_dl = ForeverDataIterator(dl)
    
    for epoch in range(config['epochs']):
        B.train()
        F.train()
        R.train()
        loss_val = 0
        mu_loss_val = 0

        for i in tqdm(range(config['iter_per_epoch']), desc=f'SHOT Epoch {epoch+1}'):            
            optimizer.zero_grad()

            images, indices = next(target_train_dl)
            images = images.to(config['device'])
            
            logits = H(F(B(images)))

            probas = F_.softmax(logits, dim=1)
            entropy_loss = entropy(probas)
            
            r_labels = np.random.randint(0, 4, config['batch'])
            r_images = rotate_batch_with_labels(images, r_labels)
            r_labels = torch.from_numpy(r_labels).to(config['device'])
            r_images = r_images.to(config['device'])

            features = F(B(images))
            r_features = F(B(r_images))

            r_outputs = R(torch.cat((features, r_features), dim=1))

            r_loss = F_.cross_entropy(r_outputs, r_labels)

            sfda_loss = r_loss + entropy_loss
            
            adv_sample = adversarial.sample.sample.detach().to(config['device'])
            adv_logits = H(F(B(adv_sample)))
            adv_labels = adv_logits.detach().clone().to(config['device'])
            adv_labels[:, config['forget_classes']] = -float('inf')
            adv_labels = F_.softmax(adv_labels, dim=1)
            mu_loss = F_.cross_entropy(adv_logits, adv_labels) 

            loss = sfda_loss + mu_loss * 10.0            
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            classifier.backbone = B
            classifier.bottleneck = F
            classifier.head = H

            # adversarial.update(deepcopy(classifier).to(config['device']))
            # Weird interaction if not copying over classifier

            loss_val += loss.item()
            mu_loss_val += mu_loss.item()

        if (not config['fast_train']) or (epoch == config['source_epochs'] - 1):
    
            # path = f"{config['save_path']}/{config['dataset']}/{config['backbone']}_{config['source']}_{config['target']}_{config['forget_classes']}_shot.pt"
            # os.makedirs(os.path.dirname(path), exist_ok=True)
            # torch.save(classifier.state_dict(), path)

            loss_val /= config['iter_per_epoch']
            source_acc = validate_shot(H, F, B, config['source_test_dl'].data_loader, config)
            target_acc = validate_shot(H, F, B, config['target_retain_test_dl'].data_loader, config)
            forget_acc = validate_shot(H, F, B, config['target_forget_dl'].data_loader, config)

            print(f"----------\n"
                f"Epoch                  : {epoch+1}\n"
                f"Train Loss             : {loss_val:.2f}\n"
                f"MU Loss                : {mu_loss_val:.2f}\n"
                f"Source Accuracy        : {source_acc:.2f}\n" 
                f"Target Accuracy        : {target_acc:.2f}\n" 
                f"Target Forget Accuracy : {forget_acc:.2f}\n"
                f"----------\n")

class FeatClassifier(nn.Module):
    def __init__(self, num_classes, bottleneck=256):
        super(FeatClassifier, self).__init__()
        self.fc = nn.Linear(bottleneck, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

def train_target_rot(classifier, config):
    """
    Trains rotation classifier for SHOT++
    """

    rotation_classifier = FeatClassifier(num_classes=4, bottleneck=2*config['bottleneck']).to(config['device'])

    for k, v in classifier.backbone.named_parameters():
        v.requires_grad = False

    for k, v in classifier.bottleneck.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in rotation_classifier.named_parameters():
        param_group += [{'params': v, 'lr': 1e-2}]

    optimizer = SGD(param_group)

    for epoch in range(config['epochs']):
        rotation_classifier.train()
        loss_val = 0
        acc = 0

        for i in tqdm(range(config['iter_per_epoch']), desc="Training Rotation Classifier"):
            images = next(config['target_retain_train_dl'])[0].to(config['device'])

            r_labels = np.random.randint(0, 4, config['batch'])
            r_images = rotate_batch_with_labels(images, r_labels)
            r_labels = torch.from_numpy(r_labels).to(config['device'])
            r_images = r_images.to(config['device'])

            features = classifier.bottleneck(classifier.backbone(images))
            r_features = classifier.bottleneck(classifier.backbone(r_images))

            r_outputs = rotation_classifier(torch.cat((features, r_features), dim=1))

            r_loss = F_.cross_entropy(r_outputs, r_labels)

            acc += (torch.argmax(r_outputs, dim=1) == r_labels).float().mean()

            loss_val += r_loss.item()

            optimizer.zero_grad()
            r_loss.backward()
            optimizer.step()

        loss_val /= config['iter_per_epoch']
        acc /= config['iter_per_epoch']

        print(f'Loss: {loss_val:.5f}\n'
              f'Acc: {acc*100:.2f}')
        
        if (not config['fast_train']) or (epoch == config['epochs'] - 1):

            path = f"{config['save_path']}/{config['dataset']}/{config['backbone']}_{config['source']}_{config['target']}_shot_rotate.pt"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(rotation_classifier.state_dict(), path)

    return rotation_classifier


def tensor_rot_90(x):
	return x.flip(2).transpose(1, 2)

def tensor_rot_180(x):
	return x.flip(2).flip(1)

def tensor_rot_270(x):
	return x.transpose(1, 2).flip(2)

def rotate_single_with_label(img, label):
	if label == 1:
		img = tensor_rot_90(img)
	elif label == 2:
		img = tensor_rot_180(img)
	elif label == 3:
		img = tensor_rot_270(img)
	return img

def rotate_batch_with_labels(batch, labels):
	images = []
	for img, label in zip(batch, labels):
		img = rotate_single_with_label(img, label)
		images.append(img.unsqueeze(0))
	return torch.cat(images)

def entropy(x):
    x = torch.clamp(x, min=1e-5)
    return -torch.sum(x * torch.log(x), dim=1).mean()



class Sample(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.sample = nn.Parameter(torch.randn(*dim), requires_grad=True)

    def forward(self):
        return self.sample
    
def validate_shot(H, F, B, dl, config):
    epoch_test_acc = 0
    with torch.no_grad():
        for (images, labels) in dl:
            images, labels = images.to(config['device']), labels.to(config['device'])
            logits = H(F(B(images)))
            epoch_test_acc += num_correct(logits, labels)
        epoch_test_acc /= len(dl.dataset)

    return 100 * epoch_test_acc