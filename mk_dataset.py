from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS


cifar = CIFAR10DVS(root='../../datasets/cifar10dvs/',train=True, data_type='frame', frames_number=4,
                         split_by='number')
