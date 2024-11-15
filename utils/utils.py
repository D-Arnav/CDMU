import matplotlib.pyplot as plt

import os

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

def log(out: str):
    """
    Dumps output to log.txt file
    """


    print(out)
    with open('log.txt', 'a') as f:
        f.write(out + '\n')


def log_fa_ra(classifier, config):
    """
    Logs Target Forget and Retain Accs
    """


    retain_acc = validate(classifier, config['target_retain_test_dl'].data_loader, config)
    forget_acc = validate(classifier, config['target_forget_dl'].data_loader, config)

    log(f"[{config['source'][0]} -> {config['target'][0]}]: {config['method']} ({retain_acc:.1f} | {forget_acc:.1f})")

    
def dump(obj, path):
    """
    Similar to pickle dump, instead takes path and creates dir if not exists
    """
    
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pickle.dump(obj, open(path, 'wb'))
    

def load(path):
    """
    Similar to pickle dump, instead takes path
    """


    obj = pickle.load(open(path, 'rb'))
    return obj


def num_correct(logits, labels):
    """
    Computes number of correctly predicted labels per batch
    """


    is_correct = (
        torch.argmax(logits, dim=1) == labels
    ).float()
    return is_correct.sum()


def validate(model, dl, config):
    """
    Computes accuracy of the model on the data in dl
    """


    model.eval()
    epoch_test_acc = 0
    with torch.no_grad():
        for (images, labels) in dl:
            images, labels = images.to(config['device']), labels.to(config['device'])
            logits = model(images)
            epoch_test_acc += num_correct(logits, labels)
        epoch_test_acc /= len(dl.dataset)

    return 100 * epoch_test_acc


def vis_adv_mu_loss(adv_hist, mu_hist, config):
    """
    Visualize the minimax game! (loss of adversarial sample and mu loss over time)
    """

    path = f"{config['vis_path']}/{config['dataset']}/{config['backbone']}/{config['source']}/{config['target']}_{config['forget_classes']}_vis_loss.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(adv_hist, label='Adversarial Loss', color='blue')
    plt.plot(mu_hist, label='Mu Loss', color='orange')
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Evolution of Adversarial and Mu Loss Over Steps', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)  

    plt.savefig(path, dpi=300, bbox_inches='tight')


# Utility Functions for UNSIR & Adversarial Minimax below
 

def create_pseudo_dl(classifier, dl, config, size=100):
    """
    Creates a pseudo dataloader with dl inputs and model predictions as outputs (hard labels)
    """

    classifier.eval()
    pseudo_inputs, pseudo_labels = [], []
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dl):
            if i > size:
                break
            inputs = inputs.to(config['device'])
            outputs = classifier(inputs)
            _, preds = torch.max(outputs, 1)
            pseudo_inputs.append(inputs.cpu())
            pseudo_labels.append(preds.cpu())
    
    pseudo_inputs = torch.cat(pseudo_inputs)
    pseudo_labels = torch.cat(pseudo_labels)
    
    pseudo_dataset = TensorDataset(pseudo_inputs, pseudo_labels)
    pseudo_dl = DataLoader(pseudo_dataset, batch_size=dl.batch_size, shuffle=True)
    
    return pseudo_dl


class Sample(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.sample = nn.Parameter(torch.randn(*dim), requires_grad=True)

    def forward(self):
        return self.sample
    

class AdversarialSample:
    def __init__(self, adv_classes, config, smooth=False):
        self.config = config
        self.adv_classes = adv_classes
        self.num_samples = self.config['num_adv']
        self.sample = Sample(self.num_samples, self.config['channels'], self.config['size'], self.config['size'])
        self.smooth = smooth
        self.lr = 1e-4
        self.mom = 0.9
        self.wt_decay = 1e-3
        self.optim = Adam(self.sample.parameters(), lr=self.lr)
        self.hist = []

   
    def learn_init(self, classifier):
        """
        Multiple steps to initially learn
        Based on implementation of Error-Maximizing Noise (Tarun. et al.)
        """
        epochs = 5
        steps = 10
        init_lr = 0.1
        init_optim = Adam(self.sample.parameters(), lr=init_lr)

        adv_labels = torch.zeros(self.num_samples, self.config['num_classes']).to(self.config['device'])
        adv_labels[:, self.adv_classes] = 1. / len(self.adv_classes)
        if self.smooth:
            adv_labels = (1 - self.config['smooth']) * adv_labels + self.config['smooth'] / self.config['num_classes']

        for epoch in range(epochs):
            classifier.eval()
            loss_val = 0
            for step in range(steps):
                sample = self.sample().to(self.config['device'])
                logits = classifier(sample)
                loss = F.cross_entropy(logits, adv_labels)

                init_optim.zero_grad()
                loss.backward()
                init_optim.step()
                loss_val += loss.item()

            loss_val /= steps
            print(f'Epoch {epoch+1} Adv Loss: {loss_val:.5f}')
        classifier.train()

    def update(self, classifier):
        """
        Single gradient descent step for update
        """

        adv_labels = torch.zeros(self.num_samples, self.config['num_classes']).to(self.config['device'])
        adv_labels[:, self.adv_classes] = 1. / len(self.adv_classes)
        if self.smooth:
            adv_labels = (1 - self.config['smooth']) * adv_labels + self.config['smooth'] / self.config['num_classes']

        sample = self.sample().to(self.config['device'])
        classifier.eval()
        logits = classifier(sample)
        loss = F.cross_entropy(logits, adv_labels)
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        self.hist.append(loss.item())

        classifier.train()


def create_noisy_dl(classifier, reg_dl, config, num_reg=float('inf')):
    """
    Creates a mixed dataloader consisting of noisy samples and regular samples
    noisy samples are generated by maximizing error (UNSIR)
    """
    
    
    EPOCHS = 5
    STEPS = 8
    LR = 0.1
    ALPHA = 0.002
    BATCH_SIZE = 32
    IDK_WHAT_THIS_IS = [1, 2, 3]

    noise = Sample(BATCH_SIZE, config['channels'], config['size'], config['size']).to(config['device'])
    optim = torch.optim.Adam(noise.parameters(), lr=LR)

    forget_labels = torch.zeros(BATCH_SIZE, config['num_classes']).to(config['device'])
    forget_labels[:, config['forget_classes']] = 1.0 / len(config['forget_classes'])

    for epoch in range(EPOCHS):
        classifier.eval()

        loss_val = 0
        for i in range(STEPS):

            images = noise()

            logits = classifier(images)
            loss = -F.cross_entropy(logits, forget_labels) + ALPHA * torch.mean(torch.sum(torch.square(images), IDK_WHAT_THIS_IS))

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_val += loss.item()

        loss_val /= STEPS

        print(f'Noise Epoch {epoch+1} Loss: {loss_val}')
    
    reg_samples = []
    n = 0
    for (images, labels) in reg_dl:
        for i in range(labels.shape[0]):
            reg_samples.append([images[i].cpu(), torch.nn.functional.one_hot(labels[i], num_classes=config['num_classes']).cpu().float()])
            n += 1
            if n >= num_reg:
                break
        if n >= num_reg:
            break
    
    noisy_samples = []
    for i in range(noise.sample.shape[0]):
        noisy_samples.append([noise().detach().cpu()[i], torch.zeros_like(forget_labels[i]).cpu().float()])

    samples = noisy_samples + reg_samples
    
    print('Noisy:',len(noisy_samples), 'Reg:', len(reg_samples), 'Total:', len(samples))
    noisy_dl = DataLoader(samples, config['batch'], shuffle=True, num_workers=8, drop_last=True)

    return noisy_dl


class Classifier_Mask(nn.Module):
    def __init__(self, classifier, emb=256):
        self.classifier = classifier
        self.em = torch.nn.Embedding(2, emb)
        self.mask = torch.empty(1, emb)

    def forward(self, x, t, s=100, all_out=False):
        if self.classifier.training:
            out, fea = self.classifier(x)
        else:
            out = self.classifier(x)


        t_ = torch.LongTensor([0]).cuda()
        self.mask = nn.Sigmoid()(self.em(t_) * s)
        t = torch.LongTensor([t]).cuda()
        mask = nn.Sigmoid()(self.em(t) * s)
        flg = torch.isnan(mask).sum()
        if flg != 0:
            print('nan occurs')
        out = out * mask

        return out, fea, self.mask