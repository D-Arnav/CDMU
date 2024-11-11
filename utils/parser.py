import argparse

def parse_args():

    parser = argparse.ArgumentParser(description='Machine Unlearning')

    parser.add_argument('--save_path', type = str, default='./weights', help='Path to save the weights')
    parser.add_argument('--dump_path', type = str, default='./dump', help='Path to save the pickle dumps')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to save the datasets')
    parser.add_argument('--vis_path', type=str, default='./vis', help='Path to save the visualizations')
    parser.add_argument('--device', type=str, default='cuda:5', help='Device to train on')
    parser.add_argument('--seed', type = int, default=1, help = 'Random seed')
    parser.add_argument('--save', action='store_true', help='Save model')
    parser.add_argument('--backbone', type=str, default='vitb16', help='Options: "vitb16", "resnet50"')
    parser.add_argument('--bottleneck', type=int, default=256, help='Bottleneck dimension')
    parser.add_argument('--fast_train', action='store_true', help='Fast training (Only validate at last epoch)')

    # Data
    parser.add_argument('--batch', type = int, default=32, help ='batch size')
    parser.add_argument('--split', type=float, default=0.8, help='train-val split for the datasets')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--source_epochs', type=int, default=10, help='Number of epochs for source model')
    parser.add_argument('--iter_per_epoch', type=int, default=100, help='Number of iterations per epoch')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of DataLoader workers')

    # Task
    parser.add_argument('-d', '--dataset', type=str, default='OfficeHome', help='Options: OfficeHome, DomainNet, Office31')
    parser.add_argument('-s', '--source', type = str, default='Product', help='Source dataset')
    parser.add_argument('-t', '--target', type = str, default='Art', help='Target dataset')

    # DA
    parser.add_argument('--da_alg', type=str, default='sfda2', help='Domain Adaptation Method')
    parser.add_argument('--smooth', type=float, default=0.1, help='Source Model Training label smoothning')
    
    # Forget Hyperparams
    parser.add_argument('--vis', action='store_true', help='Visualize Loss')
    parser.add_argument('--num_adv', type=int, default=4, help='Number of Adversarial Samples used')
    parser.add_argument('--alpha', type=float, default=5.0, help='Minimax Alpha')
    parser.add_argument('--forget_classes', type=str, default='5', help='Class to forget (Comma seperated)')
    parser.add_argument('-m', '--method', type=str, default='None', help='The forgetting algorithm')
    
    return parser.parse_args()