import sys
sys.path.append('.')

import os
import torch

import pickle

import torch

from tqdm import tqdm

from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, CenterCrop
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, DataLoader, ConcatDataset, random_split

from tllib.vision.transforms import ResizeImage
from tllib.utils.data import ForeverDataIterator

from utils.utils import dump, load


def create_loaders(config):
    """
    Adds the following loaders to config
        - Source train loader
        - Source test loader
        - Source retain train loader
        - Source retain test loader
        - Target retain train loader
        - Target retain test loader
        - Target forget loader
        - Source forget loader
        - Target retain subset loader
    """

    source_dataset = get_dataset(config['source'], config)
    target_dataset = get_dataset(config['target'], config)

    # config['forget_classes_all'] = config['forget_classes'] + config['forget_classes_2'] + config['forget_classes_3']

    target_forget_set = filter_dataset(target_dataset, config['target'], config['forget_classes'], config)
    target_retain_set = filter_dataset(target_dataset, config['target'], config['forget_classes'], config, exclude=True)
    source_forget_set = filter_dataset(source_dataset, config['source'], config['forget_classes'], config)
    source_retain_set = filter_dataset(source_dataset, config['source'], config['forget_classes'], config, exclude=True)

    target_retain_subset = get_subset(target_retain_set, config)

    source_train_dataset, source_test_dataset = random_split(source_dataset, lengths=[config['split'], 1-config['split']])
    source_retain_train_dataset, source_retain_test_dataset = random_split(source_retain_set, lengths=[config['split'], 1-config['split']])
    target_retain_train_dataset, target_retain_test_dataset = random_split(target_retain_set, lengths=[config['split'], 1-config['split']])

    config['source_train_dl'] = DataLoader(source_train_dataset, config['batch'], shuffle=True, num_workers=config['workers'], drop_last=True)
    config['source_test_dl'] = DataLoader(source_test_dataset, config['batch'], shuffle=False, num_workers=config['workers'], drop_last=True)
    config['source_retain_train_dl'] = DataLoader(source_retain_train_dataset, config['batch'], shuffle=True, num_workers=config['workers'], drop_last=True)
    config['source_retain_test_dl'] = DataLoader(source_retain_test_dataset, config['batch'], shuffle=False, num_workers=config['workers'], drop_last=True)
    config['target_retain_train_dl'] = DataLoader(target_retain_train_dataset, config['batch'], shuffle=True, num_workers=config['workers'], drop_last=True)
    config['target_retain_test_dl'] = DataLoader(target_retain_test_dataset, config['batch'], shuffle=False, num_workers=config['workers'], drop_last=True)
    config['target_forget_dl'] = DataLoader(target_forget_set, config['batch'], shuffle=True, num_workers=config['workers'], drop_last=False)
    config['source_forget_dl'] = DataLoader(source_forget_set, config['batch'], shuffle=True, num_workers=config['workers'], drop_last=False)
    config['target_retain_subset_dl'] = DataLoader(target_retain_subset, config['batch'], shuffle=True, num_workers=config['workers'], drop_last=True)

    for loader in ['source_train_dl', 'source_test_dl', 'source_retain_train_dl', 'source_retain_test_dl', 'target_retain_train_dl', 
                   'target_retain_test_dl', 'target_forget_dl', 'source_forget_dl', 'target_retain_subset_dl']:
        
        config[loader] = ForeverDataIterator(config[loader])


def get_dataset(domain: str, config: dict, size=None):
    """
    Returns dataset based on name
    """

    torch.manual_seed(config['seed'])

    transform = get_transform()

    if config['dataset'] == 'OfficeHome':
        assert domain in ['Art', 'Clipart', 'Product', 'Real_World']
        config['num_classes'] = 65
        config['size'] = 224
        config['channels'] = 3

    elif config['dataset'] == 'DomainNet':
        assert domain in ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']    
        config['num_classes'] = 345
        config['size'] = 224
        config['channels'] = 3

    elif config['dataset'] == 'Office31':
        assert domain in ['amazon', 'dslr', 'webcam']
        config['num_classes'] = 31
        config['size'] = 224
        config['channels'] = 3
    
    path = os.path.join(config['data_path'], config['dataset'], domain)        
    dataset = ImageFolder(root=path, transform=transform)
    
    if size is not None:
        dataset = Subset(dataset, torch.arange(size))
    
    return dataset


def get_transform():
    """
    Returns data transform
    """
    
    
    RESIZE = 256
    SIZE = 224
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        ResizeImage(RESIZE),
        CenterCrop(SIZE),
        ToTensor(),
        Normalize(mean=MEANS, std=STDS)
    ])

    return transform


def filter_dataset(dataset, domain, classes: list, config, exclude=False, dump=True):
    """
    Returns dataset containing samples of only certain classes.
    If exclude is True, it would exclude the classes
    """

    labels = get_labels(dataset, domain, config, dump)

    if exclude:
        mask = ~torch.isin(labels, torch.tensor(classes))
    else:
        mask = torch.isin(labels, torch.tensor(classes))

    indices = torch.nonzero(mask).squeeze().tolist()
    filtered_dataset = Subset(dataset, indices)

    return filtered_dataset


def get_labels(dataset, domain: str, config, dump_=True):
    """
    Gets the labels in order for the dataset. 
    This can be used to divide the dataset classwise.
    This data is dumped to save time
    """


    path = os.path.join(config['dump_path'], config['dataset'], domain, 'labels.p')

    if (not dump_) or (not os.path.exists(path)):
        labels = []
        for _, label in tqdm(dataset, desc=f"Filtering {config['dataset']} {domain}"):
            labels.append(label)
        labels = torch.tensor(labels)
        if dump_: dump(labels, path)

    if dump_: labels = load(path)

    return labels


def get_subset(dataset, config):
    """
    Returns the subset of data accessable after training.
    Arbitrarily considering approximately 20 samples per class
    """

    subset_size = config['num_classes'] * 20
    subset_size = min(subset_size, len(dataset))
    subset_indices = torch.randperm(len(dataset))[:subset_size]
    subset = Subset(dataset, subset_indices)

    return subset


def get_continual_loaders(forget_classes, config):
    """
    Gets loaders for the continual unlearning task.
    Retain loaders don't contain all forget classes.
    Forget loaders contain current forget classes
    Test loaders contain all forget classes
    
    T = 0 
    Forget - {1}
    Retain - C - {1}

    Forget Test - {1}
    Retain Test - C - {1}

    T = 1
    Forget - {2,3}
    Retain - C - {1,2,3}

    Forget Test - {1,2,3}
    Retain Test - C - {1,2,3}

    T = 2
    Forget - {4,5}
    Retain - C - {1,2,3,4,5}
    
    Forget Test {1,2,3,4,5}
    Retain Test C - {1,2,3,4,5}

    {1,2,3}
    {2,3}

    T Retain Train ~{1,2,3}
    T Forget {1,2,3}
    T Retain Test ~{1,2,3}
    S Test
    """

    continual_target_retain_subset_dataset = filter_dataset(config['target_retain_subset_dl'].data_loader.dataset, config['target'], forget_classes, config, exclude=True, dump=False)

    target_dataset = get_dataset(config['target'], config)
    continual_target_forget_dataset = filter_dataset(target_dataset, config['target'], forget_classes, config, exclude=False)

    continual_target_retain_test_dataset = filter_dataset(config['target_retain_test_dl'].data_loader.dataset, config['target'], forget_classes, config, exclude=True, dump=False)

    continual_source_test_dataset = filter_dataset(config['source_test_dl'].data_loader.dataset, config['source'], forget_classes, config, exclude=True, dump=False)

    config['continual_target_retain_subset_dl'] = DataLoader(continual_target_retain_subset_dataset, config['batch'], shuffle=True, num_workers=config['workers'], drop_last=True)
    config['continual_target_forget_dl'] = DataLoader(continual_target_forget_dataset, config['batch'], shuffle=True, num_workers=config['workers'], drop_last=True)
    config['continual_target_retain_test_dl'] = DataLoader(continual_target_retain_test_dataset, config['batch'], shuffle=True, num_workers=config['workers'], drop_last=True)
    config['continual_source_test_dl'] = DataLoader(continual_source_test_dataset, config['batch'], shuffle=True, num_workers=config['workers'], drop_last=True)

    for loader in ['continual_target_retain_subset_dl', 'continual_target_forget_dl', 'continual_target_retain_test_dl', 'continual_source_test_dl']:
        config[loader] = ForeverDataIterator(config[loader])

    print('Using', str(forget_classes))