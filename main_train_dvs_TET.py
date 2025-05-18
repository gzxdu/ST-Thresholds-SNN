import spikingjelly.datasets
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from data import CIFAR10Policy, Cutout
from data.sampler import DistributedSampler
from models.vgg import VGGSNN
import time
from models import *
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
import numpy as np
import myTransform
seed = 2023
import random

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def TET_loss(outputs, labels, criterion, means, lamb):
    T = outputs.size(1)
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(outputs[:, t, ...], labels)
    Loss_es = Loss_es / T # L_TET
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y) # L_mse
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd # L_Total


def build_dvscifar(args):
    transform_train = transforms.Compose([
        myTransform.ToTensor(),
        transforms.Resize(size=(48, 48)),
        transforms.RandomCrop(48, padding=4),
        transforms.RandomHorizontalFlip(),])

    transform_test = transforms.Compose([
        myTransform.ToTensor(),
        transforms.Resize(size=(48, 48))])

    train_set = CIFAR10DVS(root='../../datasets/CIFAR10-DVS', train=True, data_type='frame', frames_number=args.step, split_by='number', transform=transform_train)
    test_set = CIFAR10DVS(root='../../datasets/CIFAR10-DVS', train=False, data_type='frame', frames_number=args.step, split_by='number',  transform=transform_test)

    return train_set, test_set

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='model parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name',
                        choices=['MNIST', 'CIFAR10', 'CIFAR100'])
    parser.add_argument('--arch', default='VGG16dvs', type=str, help='dataset name',
                        choices=['CIFARNet', 'VGG16', 'res19', 'res20', 'res20m','VGG16dvs'])
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--learning_rate', default=1e-1, type=float, help='initial learning_rate')
    parser.add_argument('--epochs', default=400, type=int, help='number of training epochs')
    parser.add_argument('--thresh', default=128, type=int, help='snn threshold')
    parser.add_argument('--T', default=100, type=int, help='snn simulation length')
    parser.add_argument('--shift_snn', default=100, type=int, help='SNN left shift reference time')
    parser.add_argument('--step', default=10, type=int, help='snn step')
    parser.add_argument('--spike', action='store_true',default=True, help='use spiking network')
    parser.add_argument('--teacher', action='store_true', help='use teacher')
    parser.add_argument('--rp', action='store_true', help='use teacher')
    parser.add_argument('--recon', action='store_true', help='use teacher')
    parser.add_argument('--seed', type=int, default=666, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--weight', type=float, default=0.1, help='weight for kd loss')

    args = parser.parse_args()
    summary_dir = './vgg_dvs_TET_{}dvs_{}/summary'.format(args.dataset, str(args.step))
    model_dir = './vgg_dvs_TET_{}dvs_{}/model'.format(args.dataset, str(args.step))

    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
        os.makedirs(model_dir)
    writer = SummaryWriter(os.path.join
                            (summary_dir,args.dataset+'_'+str(args.batch_size)+'_'+args.arch+'_'+str(args.step)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    activation_save_name = args.arch + '_' + args.dataset + '_activation.npy'
    use_cifar10 = args.dataset == 'CIFAR10'

    train_dataset, val_dataset = build_dvscifar(args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=2, pin_memory=True)
    print(len(train_loader))
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=2, pin_memory=True)
    print(len(test_loader))
    best_acc = 0
    best_epoch = 0

    name = 'snn_T{}'.format(args.step) if args.spike is True else 'ann'
    model_save_name = 'raw/' + 'vgg_dvs_snn_TET_NAM.pth'

    if args.arch == 'CNN2':
        raise NotImplementedError
    elif args.arch == 'res20':
        ann = resnet20_cifar(num_classes=10 if use_cifar10 else 100, rp=args.rp)
        # ann.load_state_dict(torch.load('raw/ann_res20wd5e4.pth', map_location='cpu'))
    elif args.arch == 'res19':
        # ann = resnet19_cifar(num_classes=10 if use_cifar10 else 100, rp = args.rp)
        ann = resnet19_cifar(num_classes=10 if use_cifar10 else 100)
        # ann.load_state_dict(torch.load('raw/ann_res19.pth', map_location='cpu'))
    elif args.arch == 'res20m':
        ann = resnet20_cifar_modified(num_classes=10 if use_cifar10 else 100)
        # ann.load_state_dict(torch.load('raw/res20_ann.pth', map_location='cpu'))
    elif args.arch == 'VGG16':
        ann = vgg16_bn(num_classes=10 if use_cifar10 else 100)
        # ann.load_state_dict(torch.load('raw/ann_res20m_wd5e4.pth', map_location='cpu'))
    elif args.arch == 'VGG16dvs':
        ann = VGGSNN().to(device)
    else:
        raise NotImplementedError
    if args.spike is True:
        ann = SpikeModel(ann, args.step)
        ann.set_spike_state(True)

    ann = torch.nn.DataParallel(ann).to(device)
    num_epochs = 400
    criterion = nn.CrossEntropyLoss()
    all_params = ann.parameters()
    my_params = []
    choice_param_name = ['my_NAM', 'temporal', 'relu']
    for pname, p in ann.named_parameters():
        if 'temporal' in pname:
            print(pname, p.shape)
            my_params.append(p)
    params_id = list(map(id, my_params))
    other_params = list(filter(lambda p: id(p) not in params_id, all_params))
    optimizer = torch.optim.SGD([
        {'params': other_params},
        {'params': my_params,'weight_decay': 0.},
    ],
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs)
    print(ann)


    correct = torch.Tensor([0.]).to(device)
    total = torch.Tensor([0.]).to(device)
    acc = torch.Tensor([0.]).to(device)
    ann.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = ann(inputs).mean(1)
            # embed()
            _, predicted = outputs.cpu().max(1)
            total += (targets.size(0))
            correct += (predicted.eq(targets.cpu()).sum().item())

    acc = 100 * correct / total
    print('Test Accuracy of the model on the 10000 test images: {}'.format(acc.item()))

    for epoch in range(num_epochs):
        running_loss = 0
        start_time = time.time()
        ann.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            labels = labels.to(device)
            images = images.float().to(device)
            outputs = ann(images)
            loss = TET_loss(outputs,labels,criterion,1,1e-3)

            if (i + 1) % 40 == 0:
                print("Loss: ", loss.item())
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
            model_save_name = (os.path.join
                               (model_dir, args.dataset + '_' + str(args.batch_size) + '_' +
                                args.arch + '_step_' + str(args.step) + '_epoch_' + str(epoch) + '_' + str(
                                   acc.item())) + '.pth')
            torch.save(ann.state_dict(), model_save_name)
        print('best_acc is: {}'.format(best_acc.item()))
        print('best_iter: {}'.format(best_epoch))
        print('Iters: {}\n\n'.format(epoch))
        writer.add_scalar('Test Acc /epoch', 100. * correct / len(test_loader.dataset), epoch)

    writer.close()