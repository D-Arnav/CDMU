import pickle
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import models

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from utils.utils import num_correct, validate, dump, load, AdversarialSample
from utils.sfda import SFDA2
from utils.loss import CrossEntropyLabelSmooth

import torch
# import logging
import sklearn
import sklearn.metrics
from torch.autograd import Variable
# from utils.utils import parse_address

# from .data_list import ImageList
import torch.utils.data as util_data
from torchvision import transforms
# from datasets import sampler
# from datasets.sampler import N_Way_K_Shot_BatchSampler, TaskSampler, PseudoWeightedRandomSampler
from collections import Counter
import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


def get_dataloader_from_image_filepath(
        images_file_path,  batch_size=32, resize_size=256, is_train=True, crop_size=224, center_crop=True, args=None,
                     sampler=None, rand_aug_size=0, is_source=True):

    if images_file_path is None:
        return None, None

    data_sampler = None
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train is not True:  # eval mode
        transformer = transforms.Compose([
            ResizeImage(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize])

        images = ImageList(open(images_file_path).readlines(), transform=transformer, args=args)
        images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    else:  # training mode
        if center_crop:
            transformer = transforms.Compose([ResizeImage(resize_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.CenterCrop(crop_size),
                                              transforms.ToTensor(),
                                              normalize])
        else:
            transformer = transforms.Compose([ResizeImage(resize_size),
                                              transforms.RandomCrop(crop_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize])

        images = ImageList(open(images_file_path).readlines(), transform=transformer, args=args, rand_aug_size=rand_aug_size)

        if sampler is None:
            images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
        elif sampler is not None:
            images_loader, data_sampler = select_training_dataloader(images, args, sampler, is_source)
        else:
            raise ValueError('could not create dataloader under the given config')

    return images_loader, data_sampler

def select_training_dataloader(images, args, sampler, is_source):
    if sampler == 'N_Way_K_Shot_BatchSampler':
        images_loader, data_sampler = nway_kshot_dataloader(images, args)
    elif sampler == 'ClassBalancedBatchSampler':
        images_loader, data_sampler = class_balanced_dataloader(images, args, is_source)
    else:
        images_loader, data_sampler = self_training_dataloader(images, args, sampler)
    return images_loader, data_sampler

def class_balanced_dataloader(images, args, is_source):
    if is_source:
        # use ground truth-label
        count_dict = Counter(images.labels)
        count_dict_full = {lbl: 0 for lbl in range(args.class_num)}
        for k, v in count_dict.items(): count_dict_full[k] = v

        count_dict_sorted = {k: v for k, v in sorted(count_dict_full.items(), key=lambda item: item[0])}
        class_sample_count = np.array(list(count_dict_sorted.values()))
        class_sample_count = class_sample_count / class_sample_count.max()
        class_sample_count += 1e-8

        weights = 1 / torch.Tensor(class_sample_count)
        sample_weights = [weights[l] for l in images.labels]
        sample_weights = torch.DoubleTensor(np.array(sample_weights))
        class_balanced_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        images_loader = util_data.DataLoader(images, sampler=class_balanced_sampler, \
                                                   batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
    else:
        # use pseudo label
        sample_weights = torch.DoubleTensor(np.ones(len(images.labels)))
        class_balanced_sampler = PseudoWeightedRandomSampler(sample_weights, len(images.labels))
        images_loader = util_data.DataLoader(images, sampler=class_balanced_sampler, \
                                             batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)

    return images_loader, class_balanced_sampler


def nway_kshot_dataloader(images, args):
    task_sampler = TaskSampler(set(images.labels), args)
    n_way_k_shot_sampler = N_Way_K_Shot_BatchSampler(images.labels, args.train_steps, task_sampler)
    meta_loader = util_data.DataLoader(images, shuffle=False, batch_sampler=n_way_k_shot_sampler, num_workers=args.num_workers)
    return meta_loader, n_way_k_shot_sampler

def self_training_dataloader(images, args, _sampler):
    task_sampler = TaskSampler(set(images.labels), args)
    self_train_sampler_cls = getattr(sampler, _sampler)
    self_train_sampler = self_train_sampler_cls(args.train_steps, task_sampler, args)
    self_train_dataloader = util_data.DataLoader(images, shuffle=False, batch_sampler=self_train_sampler, num_workers=args.num_workers)
    return self_train_dataloader, self_train_sampler


class DataLoaderManager:
    def __init__(self, args):
        self.args = args
        self.train_source_loader, self.train_source_sampler = get_dataloader_from_image_filepath(
            args.source_path, args=args, batch_size=args.batch_size,
            sampler=args.train_source_sampler, center_crop=args.center_crop, rand_aug_size=args.source_rand_aug_size, is_source=True)
        self.train_target_loader, self.train_target_sampler = get_dataloader_from_image_filepath(
                args.target_path, args=args, batch_size=args.batch_size,
            sampler=args.train_target_sampler, center_crop=args.center_crop, rand_aug_size=args.target_rand_aug_size, is_source=False)

        self.val_source_loader, self.val_source_sampler = get_dataloader_from_image_filepath(
            args.source_path, args=args, batch_size=args.batch_size, is_train=False, is_source=True)
        self.val_target_loader, self.val_target_sampler = get_dataloader_from_image_filepath(
            args.target_path, args=args, batch_size=args.batch_size, is_train=False, is_source=False)

        if type(args.test_path) is list:
            tst_ls = []
            for tst_addr in args.test_path:
                tst_ls.append(list(get_dataloader_from_image_filepath(
                tst_addr, args=args, batch_size=args.batch_size, is_train=False, is_source=False)))
            self.test_loader, self.test_sampler = zip(*tst_ls)
        else:
            self.test_loader, self.test_sampler = get_dataloader_from_image_filepath(
            args.test_path, args=args, batch_size=args.batch_size, is_train=False, is_source=False)

    def update_sampler(self, model_instance, iter):
        if self.train_target_sampler is not None and hasattr(self.train_target_sampler, 'update'):
            self.train_target_sampler.update(model_instance, self.val_target_loader, iter)

        if self.train_source_sampler is not None and hasattr(self.train_source_sampler, 'update'):
            self.train_source_sampler.update(model_instance, self.val_source_loader, iter)


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter([])

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)
        
class Base_cls(nn.Module):
    def __init__(self, dim=1024, class_num=31, use_hidden_layer=False, l2_normalize=False, temperature=1.0, use_dropout=False):
        super(Base_cls, self).__init__()
        if use_hidden_layer:
            self.classifier_layer_list = [nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.5), nn.Linear(dim, class_num)] \
                if use_dropout else [nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, class_num)]
        else:
            self.classifier_layer_list = [nn.Linear(dim, class_num)]

        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)

        self.l2_normalize = l2_normalize
        self.temperature = temperature

        for layer in self.classifier_layer:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, 0.01)
                layer.bias.data.fill_(0.0)

        self.parameter_list = [{"params": self.classifier_layer.parameters(), "lr": 1}]

    def forward(self, features):
        if not self.l2_normalize:
            score = self.classifier_layer(features) / self.temperature
        else:
            if isinstance(self.classifier_layer, nn.Sequential):
                features = self.classifier_layer[:-1](features)
                features = F.normalize(features)
                score = self.classifier_layer[-1](features) / self.temperature
            else:
                features = F.normalize(features)
                score = self.classifier_layer(features) / self.temperature
        return score


class Base_feat(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=1024, use_dropout=False):
        super(Base_feat, self).__init__()
        self.base_network = ResNet50Fc()
        self.use_bottleneck = use_bottleneck
        self.use_dropout = use_dropout

        self.parameter_list = [{"params": self.base_network.parameters(), "lr": 0.1}]

        if self.use_bottleneck:
            self.bottleneck_layer_list = [nn.Linear(self.base_network.output_num(), bottleneck_dim),
                                          nn.BatchNorm1d(bottleneck_dim), nn.ReLU()]
            if self.use_dropout:
                self.bottleneck_layer_list.append(nn.Dropout(0.5))
            self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)

            self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
            self.bottleneck_layer[0].bias.data.fill_(0.1)

            self.parameter_list.append({"params":self.bottleneck_layer.parameters(), "lr":1})


    def freeze_param(self, freeze=True):
        for param in self.base_network.parameters():
            param.requires_grad = not freeze
        if self.use_bottleneck:
            for param in self.bottleneck_layer.parameters():
                param.requires_grad = not freeze

    def forward(self, inputs):
        features = self.base_network(inputs)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        return features

class ResNet50Fc(nn.Module):
    def __init__(self):
        super(ResNet50Fc, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.__in_features = model_resnet50.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class Base(object):
    def __init__(self, base_net='ResNet50', bottleneck_dim=1024, class_num=31, use_bottleneck=True, use_gpu=True, args=None):
        self.c_net_feat = Base_feat(base_net, use_bottleneck, bottleneck_dim, use_dropout=False)
        dim = bottleneck_dim if use_bottleneck else self.c_net_feat.base_network.output_num()
        self.c_net_cls = Base_cls(dim, class_num, use_hidden_layer=False, use_dropout=False,
                        l2_normalize=True, temperature=0.05)

        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num

        if self.use_gpu:
            self.c_net_feat = self.c_net_feat.cuda()
            self.c_net_cls = self.c_net_cls.cuda()

    def to_dicts(self):
        return [self.c_net_feat.state_dict(), self.c_net_cls.state_dict()]

    def from_dicts(self, dicts):
        self.c_net_feat.load_state_dict(dicts[0])
        self.c_net_cls.load_state_dict(dicts[1])

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        features = self.c_net_feat(inputs)
        outputs = self.c_net_cls(features)
        return outputs

    def predict(self, inputs, output='prob'):
        features = self.c_net_feat(inputs)
        outputs = self.c_net_cls(features)
        if output == 'prob':
            softmax_outputs = F.softmax(outputs)
            return softmax_outputs
        elif output == 'score':
            return outputs
        elif output == 'score+feature':
            return outputs, features
        elif output == 'feature':
            return features
        else:
            raise NotImplementedError('Invalid output')

    def get_parameter_list(self):
        return self.c_net_feat.parameter_list + self.c_net_cls.parameter_list

    def set_train(self, mode):
        self.c_net_feat.train(mode)
        self.c_net_cls.train(mode)
        self.is_train = mode

def load_checkpoint(model, filename):
    with open(filename, "rb" ) as fc:
        dicts = pickle.load(fc)
    try:
        model.from_dicts(dicts)
    except:
        new_dicts = []
        for _dict in dicts:
            new_dict = {}
            if isinstance(_dict, OrderedDict):
                for name, param in _dict.items():
                    namel = name.split('.')
                    key = '.'.join(namel[1:])
                    new_dict.update({key: param})
                new_dicts.append(new_dict)
            else:
                new_dicts.append(_dict)
        model.from_dicts(new_dicts)


def train_src(config):
    model = Base(bottleneck_dim=2048, class_num=65)

    
    optimizer = SGD(model.get_parameter_list(),
                    lr=1e-2, momentum=0.9, weight_decay=1e-3, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: 1e-2 * (1. + 1e-3 * float(x)) ** -0.75)

    for epoch in range(5):
        model.set_train(True)
        loss_val = 0.0

        for i in tqdm(range(config['iter_per_epoch']), desc=f"Epoch {epoch+1}"):
            images, labels = next(config['source_train_dl'])
            images, labels = images.to(config['device']), labels.to(config['device'])

            optimizer.zero_grad()
            logits = model.forward(images)

            loss = CrossEntropyLabelSmooth(config)(logits, labels)

            loss_val += loss.item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        loss_val /= config['iter_per_epoch']
        

        with torch.no_grad():
            model.set_train(False)
            acc = 0
            for images, labels in config['source_test_dl'].data_loader:
                images, labels = images.to(config['device']), labels.to(config['device'])
                logits = model.predict(images)
                acc += (torch.argmax(logits, dim=1) == labels).float().sum()
            acc /= len(config['source_train_dl'].data_loader.dataset)

        print(f"----------\n"
                f"Loss     : {loss_val:.2f}\n"
                f"Accuracy : {acc*100:.2f}\n" 
                f"----------\n")
    return model
    
def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)


class ImageList(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=default_loader, args=None, rand_aug_size=0):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of"))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.labels = [label for (_, label) in imgs]
        self.rand_aug_size = rand_aug_size
        if self.rand_aug_size > 0:
            self.rand_aug_transform = copy.deepcopy(self.transform)
            self.rand_aug_transform.transforms.insert(0, RandAugment(args.rand_aug_n, args.rand_aug_m))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img_ = self.loader(path)
        if self.transform is not None:
            img = self.transform(img_)
        if self.target_transform is not None:
            target = self.target_transform(target)

        rand_imgs = [self.rand_aug_transform(img_) for _ in range(self.rand_aug_size)]
        return img, target, index, rand_imgs

    def __len__(self):
        return len(self.imgs)


def train_apa(config):    

    model = Base(bottleneck_dim=2048, class_num=65)
    
    load_checkpoint(model, 'weights/RW_base.pth')

    args = Args(config)
    data_loader_manager = DataLoaderManager(args)
    train_source_loader, train_target_loader = data_loader_manager.train_source_loader, data_loader_manager.train_target_loader
    val_source_loader, val_target_loader = data_loader_manager.val_source_loader, data_loader_manager.val_target_loader

    res = evaluate(model, val_source_loader, 'source')

    print(res['accuracy'])
    return

    model = train_src(config)

    vat_iter = 1
    vat_xi = 10
    vat_eps = 30
    vat_alpha = 0.1
    st_conf = 0.75

    optimizer = SGD(model.get_parameter_list(), lr=1e-2, momentum=0.9, weight_decay=5e-5, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: 1e-2 * (1. + 1e-4 * float(x)) ** -0.75)

    # adversarial = AdversarialSample(config['forget_classes'], config)
    # adversarial.learn_init(source_classifier)
    
    for epoch in range(config['epochs']):
        model.set_train(True)
        loss_val = 0
        
        for i in tqdm(range(config['iter_per_epoch']), desc=f"Epoch {epoch+1}"):
            images, indices = next(config['target_retain_train_dl'])
            images = images.to(config['device'])
            
            features = model.c_net_feat(images)
            logits = model.c_net_cls(features)
            
            # ST

            confs, pseudo_labels = F.softmax(logits, dim=1).max(dim=1)

            st_loss = (F.cross_entropy(logits, pseudo_labels, reduction='none') * (confs >= st_conf).float().detach()).mean()

            # VAT

            d = torch.rand(features.shape).sub(0.5).to(config['device'])
            outputs = F.softmax(logits, dim=1).detach().clone()

            for i in range(vat_iter):
                d = vat_xi * l2_normalize(d)
                d.requires_grad_()
                outputs_ = model.c_net_cls(features.detach() + d)
                logp_hat = F.log_softmax(outputs_, dim=1)
                adv_dist = F.kl_div(logp_hat, outputs, reduction='batchmean')
                adv_dist.backward()
                d = l2_normalize(d.grad)
                model.c_net_cls.zero_grad()

            r_adv = d * vat_eps
            act = features * r_adv.detach().clone()

            outputs_ = model.c_net_cls(act)
            logp_hat = F.log_softmax(outputs_, dim=1)
            adap_loss = F.kl_div(logp_hat, outputs, reduction='batchmean')

            loss = st_loss + adap_loss * vat_alpha
            
            # sample = adversarial.sample().detach().to(config['device'])
            # features = model.c_net_feat(sample)
            # logits = model.c_net_cls(features)
            # labels = logits.detach().clone().to(config['device'])
            # labels[:, config['forget_classes']] = -float('inf')
            # labels = F.softmax(labels, dim=1)
            # outputs = F.softmax(logits, dim=1)
            # mu_loss = F.cross_entropy(outputs, labels)
            
            # loss += mu_loss * 0.01

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            loss_val += loss.item()
            # adversarial.update(source_classifier)

        if (not config['fast_train']) or (epoch == config['epochs'] - 1):
        
            # if save:
            #     path = f"{config['save_path']}/{config['dataset']}/{config['backbone']}_{config['source']}_{config['target']}_{config['forget_classes']}_{save}.pt"
            #     os.makedirs(os.path.dirname(path), exist_ok=True)
            #     torch.save(classifier.state_dict(), path)

            loss_val /= config['iter_per_epoch']
            res = evaluate(model, config['target_forget_dl'].data_loader, 'target')
            res2 = evaluate(model, config['target_retain_test_dl'].data_loader, 'target')

            print(f"----------\n"
                f"Epoch                  : {epoch+1}\n"
                f"Train Loss             : {loss_val:.2f}\n"
                f"Target Accuracy        : {res2['accuracy']:.2f}\n" 
                f"Target Forget Accuracy : {res['accuracy']:.2f}\n"
                f"----------\n")

    return classifier


def l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

def evaluate(model_instance, input_loader, domain):
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True

    with torch.no_grad():
        for i in range(num_iter):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            if model_instance.use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            probabilities = model_instance.predict(inputs)

            probabilities = probabilities.data.float()
            labels = labels.data.float()
            if first_test:
                all_probs = probabilities
                all_labels = labels
                first_test = False
            else:
                all_probs = torch.cat((all_probs, probabilities), 0)
                all_labels = torch.cat((all_labels, labels), 0)

    _, predict = torch.max(all_probs, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_labels) / float(all_labels.size()[0])

    # class-based average accuracy
    avg_acc = sklearn.metrics.balanced_accuracy_score(all_labels.cpu().numpy(),
                                                      torch.squeeze(predict).float().cpu().numpy())

    cm = sklearn.metrics.confusion_matrix(all_labels.cpu().numpy(),
                                          torch.squeeze(predict).float().cpu().numpy())
    accuracies = cm.diagonal() / cm.sum(1)
    precisions = cm.diagonal() / (cm.sum(0)+1e-6)
    predictions = cm.sum(0)

    f1s = 2/(1/(accuracies+1e-6) + 1/(precisions+1e-6))

    model_instance.set_train(ori_train_state)
    return {'accuracy': accuracy, 'per_class_accuracy': avg_acc, 'accuracies': accuracies, 'precisions':precisions, 'predictions':predictions, 'f1s':f1s}

class Args:
    def __init__(self, config):
        self.num_workers = 2
        self.class_num = config['num_classes']
        self.batch_size = config['batch']
        self.source_path = f"{config['data_path']}/{config['dataset']}/{config['source']}"
        self.target_path = f"{config['data_path']}/{config['dataset']}/{config['target']}"
        self.train_source_sampler = None
        self.train_target_sampler = "ClassBalancedBatchSampler"
        self.center_crop = False
        self.source_rand_aug_size = 0
        self.target_rand_aug_size = 0
        self.rand_aug_m = 2.0
        self.rand_aug_n = 3
        self.test_path = None

