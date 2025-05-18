import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from data import CIFAR10Policy, Cutout
from data.sampler import DistributedSampler
import time
from models import *
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np

seed = 2023
import random
random.seed(2023)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def build_data(batch_size=128, cutout=False, workers=4, use_cifar10=False, auto_aug=False):

    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if auto_aug:
        aug.append(CIFAR10Policy())

    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(Cutout(n_holes=1, length=16))

    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root='../../datasets/cifar10/',
                                train=True, download=True, transform=transform_train)
        val_dataset = CIFAR10(root='../../datasets/cifar10/',
                              train=False, download=True, transform=transform_test)

    else:
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root='../../datasets/cifar100/',
                                 train=True, download=True, transform=transform_train)
        val_dataset = CIFAR100(root='../../datasets/cifar100/',
                               train=False, download=False, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=workers, pin_memory=True)

    return train_loader, val_loader


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='model parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name',
                        choices=['MNIST', 'CIFAR10', 'CIFAR100'])
    parser.add_argument('--arch', default='res19', type=str, help='dataset name',
                        choices=['CIFARNet', 'VGG16', 'res19', 'res20', 'res20m','res18'])
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--learning_rate', default=1e-1, type=float, help='initial learning_rate')
    parser.add_argument('--epochs', default=400, type=int, help='number of training epochs')
    parser.add_argument('--thresh', default=128, type=int, help='snn threshold')
    parser.add_argument('--T', default=100, type=int, help='snn simulation length')
    parser.add_argument('--shift_snn', default=100, type=int, help='SNN left shift reference time')
    parser.add_argument('--step', default=4, type=int, help='snn step')
    parser.add_argument('--spike', action='store_true',default=True, help='use spiking network')
    parser.add_argument('--teacher', action='store_true', help='use teacher')
    parser.add_argument('--rp', action='store_true', help='use teacher')
    parser.add_argument('--recon', action='store_true', help='use teacher')
    parser.add_argument('--seed', type=int, default=2023, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--weight', type=float, default=0.1, help='weight for kd loss')

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    summary_dir = './ablation_epoch/resnet19_NAM_{}_{}_{}/summary'.format(args.dataset, str(args.step),str(args.epochs))
    model_dir = './ablation_epoch/resnet19_NAM_{}_{}_{}/model'.format(args.dataset, str(args.step),str(args.epochs))

    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
        os.makedirs(model_dir)
    writer = SummaryWriter(os.path.join
                            (summary_dir,args.dataset+'_'+str(args.batch_size)+'_'+args.arch+'_'+str(args.step)))


    batch_size = args.batch_size
    learning_rate = args.learning_rate
    use_cifar10 = args.dataset == 'CIFAR10'

    train_loader, test_loader = build_data(cutout=True, use_cifar10=use_cifar10, auto_aug=True, batch_size=args.batch_size)
    best_acc = 0
    best_epoch = 0

    name = 'snn_T{}'.format(args.step) if args.spike is True else 'ann'

    if args.arch == 'CNN2':
        raise NotImplementedError
    elif args.arch == 'res20':
        ann = resnet20_cifar(num_classes=10 if use_cifar10 else 100)
    elif args.arch == 'res19':
        ann = resnet19_cifar(num_classes=10 if use_cifar10 else 100)
    elif args.arch == 'res20m':
        ann = resnet20_cifar_modified(num_classes=10 if use_cifar10 else 100)
    elif args.arch == 'VGG16':
        ann = vgg16_bn(num_classes=10 if use_cifar10 else 100)
    else:
        raise NotImplementedError
    if args.spike is True:
        ann = SpikeModel(ann, args.step)
        ann.set_spike_state(True)


    ann.to(device)
    num_epochs = 400
    criterion = nn.CrossEntropyLoss().to(device)
    all_params = ann.parameters()
    optimizer = torch.optim.SGD(ann.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs)
    print(ann)
    #
    total = sum([param.nelement() for param in ann.parameters()])
    print('Number of parameter: % .4fM' % (total / 1e6))

    correct = torch.Tensor([0.]).to(device)
    total = torch.Tensor([0.]).to(device)
    acc = torch.Tensor([0.]).to(device)
    ann.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = ann(inputs).mean(1)
            #embed()
            _, predicted = outputs.cpu().max(1)
            total += (targets.size(0))
            correct += (predicted.eq(targets.cpu()).sum().item())

    acc = 100 * correct / total
    print('Test Accuracy of the model on the 10000 test images: {}'.format(acc.item()))
    writer.add_scalar('first Acc /epoch', 100. * correct / len(test_loader.dataset))

    
    for epoch in range(num_epochs):
        running_loss = 0
        start_time = time.time()
        ann.train()

        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            labels = labels.to(device)
            images = images.float().to(device)
            outputs = ann(images).mean(1)

            loss = criterion(outputs, labels)
            
            if (i + 1) % 40 == 0:
                print("Loss: ",loss.item())
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (i + 1) % 40 == 0:
                print('Time elapsed: {}'.format(time.time() - start_time))
                writer.add_scalar('Train Loss /batchidx', loss, i + len(train_loader) * epoch)
        scheduler.step()

        correct = torch.Tensor([0.]).to(device)
        total = torch.Tensor([0.]).to(device)
        acc = torch.Tensor([0.]).to(device)

        ann.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = ann(inputs).mean(1)
                _, predicted = outputs.cpu().max(1)
                total += (targets.size(0))
                correct += (predicted.eq(targets.cpu()).sum().item())

        acc = 100 * correct / total
        print('Test Accuracy of the model on the 10000 test images: {}'.format(acc.item()))
        if best_acc < acc:
            best_acc = acc
            best_epoch = epoch + 1
            model_save_name=(os.path.join
                               (model_dir,args.dataset+'_'+str(args.batch_size)+'_'+
                                args.arch+'_step_'+str(args.step)+'_epoch_'+str(epoch)+'_'+str(acc.item()))+'.pth')
            torch.save(ann.state_dict(), model_save_name)
        if (best_acc-acc).item()>30:
            model_save_name=(os.path.join
                               (model_dir,args.dataset+'_'+str(args.batch_size)+'_'+
                                args.arch+'_step_'+str(args.step)+'_epoch_'+str(epoch)+'_'+str(acc.item()))+'.pth')
            torch.save(ann.state_dict(), model_save_name)
        print('best_acc is: {}'.format(best_acc.item()))
        print('best_iter: {}'.format(best_epoch))
        print('Iters: {}\n\n'.format(epoch))
        writer.add_scalar('Test Acc /epoch', 100. * correct / len(test_loader.dataset), epoch)
        
    writer.close()
