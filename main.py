from copy import deepcopy

import os

import timm

from tllib.alignment.dann import ImageClassifier

import torch
import torch.nn as nn

from utils.data import create_loaders
from utils.train import train_source, train_sfda2_old, train_g_sfda
from utils.forget import sample_minimax_old, finetune_old, unsir, sample_minimax_continual
from utils.utils import log_fa_ra, log, validate
from utils.parser import parse_args
from utils.apa import train_apa
from utils.shot import train_shot, train_forget_shot

import warnings

warnings.filterwarnings("ignore") # Supress CUDNN

if not torch.cuda.is_available():
    print("Warning! cuda device not found")

args = parse_args()

config = {
    'seed': args.seed,
    'split': args.split,
    'batch': args.batch,
    'bottleneck': args.bottleneck,
    'workers': args.num_workers,
    'epochs': args.epochs,
    'source_epochs': args.source_epochs,
    'smooth': args.smooth,
    'iter_per_epoch': args.iter_per_epoch,
    'save': args.save,
    'device': args.device,
    'backbone': args.backbone,
    'dataset': args.dataset,
    'source': args.source,
    'target': args.target,
    'data_path': args.data_path,
    'dump_path': args.dump_path,
    'save_path': args.save_path,
    'vis_path': args.vis_path,
    'fast_train': args.fast_train,
    'method': args.method,
    'num_adv': args.num_adv,
    'minimax_alpha': args.alpha,
    'vis': args.vis,
    'forget_classes': [1],
    'forget_classes_2': [2,3],
    'forget_classes_3': [4,5],
    'finetune_epochs': 2,
}

torch.manual_seed(config['seed'])

create_loaders(config)

if config['backbone'] == 'vitb16':
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.out_features = model.head.in_features
    model.head = nn.Identity()

    pool_layer = torch.nn.Identity()
    classifier = ImageClassifier(model, config['num_classes'], pool_layer=pool_layer, bottleneck_dim=config['bottleneck']).to(config['device'])



source_path = f"{config['save_path']}/{config['dataset']}/{config['backbone']}_{config['source']}.pt"

if os.path.exists(source_path):
    source_classifier = deepcopy(classifier)
else:
    source_classifier = train_source(deepcopy(classifier), config)

source_classifier.load_state_dict(torch.load(source_path, map_location=config['device']))



if config['method'] == 'original':
    target_classifier = train_shot(source_classifier, config)

    # target_classifier = sfda_apa_u(source_classifier, config)
    # target_classifier = train_sfda2_old(source_classifier, config, save="original")


if config['method'] == 'retrain':
    source_retain_classifier = train_source(deepcopy(classifier), config, retain=True, save=False, smooth=False)
    target_classifier = train_shot(source_retain_classifier, config)

if config['method'] == 'finetune':
    target_classifier = train_shot(source_classifier, config)
    config['epochs'] = 5
    config['iter_per_epoch'] = 100
    target_classifier = train_shot(target_classifier, config)

if config['method'] == 'minimax':
    target_classifier = train_forget_shot(source_classifier, config)


if config['method'] == 'unsir':
    source_classifier = unsir(source_classifier, config)
    target_classifier = train_forget_shot(source_classifier, config)

if config['method'] == 'minimax_continual':
    target_classifier = sample_minimax_old(source_classifier, config, vis=False)
    
    retain_acc = validate(target_classifier, config['target_retain_test_dl'].data_loader, config)
    forget_acc = validate(target_classifier, config['target_forget_dl'].data_loader, config)
    log(f"[{config['source'][0]} -> {config['target'][0]}]: {config['method']} T1 ({retain_acc:.1f} | {forget_acc:.1f})")

    config['minimax_alpha'] = 15.0
    forget_classes_all = config['forget_classes'] + config['forget_classes_2']
    forget_classes = config['forget_classes_2']
    target_classifier = sample_minimax_continual(target_classifier, forget_classes, forget_classes_all, config)
        
    retain_acc = validate(target_classifier, config['continual_target_retain_test_dl'].data_loader, config)
    forget_acc = validate(target_classifier, config['continual_target_forget_dl'].data_loader, config)
    log(f"[{config['source'][0]} -> {config['target'][0]}]: {config['method']} T2 ({retain_acc:.1f} | {forget_acc:.1f})")

    forget_classes_all = config['forget_classes'] + config['forget_classes_2'] + config['forget_classes_3']
    forget_classes = config['forget_classes_3']
    target_classifier = sample_minimax_continual(target_classifier, forget_classes, forget_classes_all, config)
    
    retain_acc = validate(target_classifier, config['continual_target_retain_test_dl'].data_loader, config) 
    forget_acc = validate(target_classifier, config['continual_target_forget_dl'].data_loader, config)
    log(f"[{config['source'][0]} -> {config['target'][0]}]: {config['method']} T3 ({retain_acc:.1f} | {forget_acc:.1f})")

log_fa_ra(target_classifier, config)


# Might want to implement black box MIA scores

# https://github.com/vikram2000b/bad-teaching-unlearning/blob/main/metrics.py
# Step 1: Consider the forget class is class #5
# Step 2: Get Model trained on our method as: 
#           - Source trained on All 65 classes
#           - Target trained on 64 classes -- all except class 5
#           - Method performed with forget class 5
# Step 3: Train a shadow model on subset of retain set and forget set
# Step 4: Train an attacker model (binary classifier) on confidence scores of retain and forget samples. 
# Step 5: Test this model on confidence scores of Original, Retrain, Finetune, Ours, Bad Teacher 

# torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 29.06 GiB. GPU 4 has a total capacity of 31.74 GiB of which 9.37 GiB is free. 
# Including non-PyTorch memory, this process has 22.37 GiB memory in use. Of the allocated memory 17.89 GiB is allocated by PyTorch, 
# and 3.32 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try 
# setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
