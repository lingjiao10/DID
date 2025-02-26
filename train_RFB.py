from __future__ import print_function
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.init as init
import argparse
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
from data import VOCroot, COCOroot, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300, AnnotationTransform, \
    COCODetection, VOCDetection, detection_collate, BaseTransform, preproc, preproc_tf, Visdom
from layers.modules import MultiBoxLoss, MultiBoxLoss_tf_source, MultiBoxLoss_tf_source_combine
from layers.functions import PriorBox
import time
from collections import namedtuple

# used by trainer
from utils.vis_tool import Visualizer
from torchnet.meter import ConfusionMeter, AverageValueMeter
from utils import array_tool as at

import math

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"
# torch.cuda.set_device(4)

LossTuple = namedtuple('LossTuple',
                       ['loc_loss',
                        'conf_loss',
                        'bin_loss',
                        'total_loss'
                        ])


#-----------------trainer-------------------
class MyTrainer(nn.Module):
    def __init__(self):
        super(MyTrainer, self).__init__()

        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def update_meters(self, losses):
            loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
            for key, meter in self.meters.items():
                meter.add(loss_d[key])

    def get_meter_data(self):
            return {k: v.value()[0] for k, v in self.meters.items()}

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
#-------------------------------------------

parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')
parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument(
    '--basenet', default='./weights/vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=8,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=4, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate',
                    default=4e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
# parser.add_argument(
#     '--resume_net', default='./weights/task1-source/tradictionvoc/RFB_vgg_VOC_epoches_70.pth', help='resume net for retraining')

parser.add_argument(
    '--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=70,
                    type=int, help='resume iter for retraining')
parser.add_argument('-max', '--max_epoch', default=300,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1,
                    type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True,
                    type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='./weights/task1-source/tradictionvoc/',
                    help='Location to save checkpoint models')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.dataset == 'VOC':
    #train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    #train_sets = [('2007', 'trainval')]
    #train_sets = [('2007-task3-source', 'trainval')]
    #train_sets = [('2007-task3-target1', 'trainval')]
    train_sets = [('2017-task1-source', 'train')]
    # train_sets = [('2017-task1-source-check', 'train')]
    cfg = (VOC_300, VOC_512)[args.size == '512']
else:
    # train_sets = [('2014', 'train'),('2014', 'valminusminival')]
    train_sets = [('2017', 'train')]
    # train_sets = [('2014', 'valminusminival')]
    cfg = (COCO_300, COCO_512)[args.size == '512']

cfg = VOC_300

if args.version == 'RFB_vgg':
    from models.RFB_Net_vgg import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_net

    cfg = COCO_mobile_300
else:
    print('Unkown version!')

img_dim = (300, 512)[args.size == '512']
rgb_means = ((104, 117, 123), (103.94, 116.78, 123.68))[args.version == 'RFB_mobile']
p = (0.6, 0.2)[args.version == 'RFB_mobile']
num_classes = (61, 81)[args.dataset == 'COCO']
batch_size = args.batch_size
weight_decay = 0.0005
gamma = 0.1
momentum = 0.9

net = build_net('train', img_dim, num_classes)
print(net)
if args.resume_net == None:
    base_weights = torch.load(args.basenet)
    print('Loading base network...')
    net.base.load_state_dict(base_weights)

    def xavier(param):
        init.xavier_uniform(param)

    def weights_init(m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0


    print('Initializing weights...')
    # initialize newly added layers' weights with kaiming_normal method
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)
    net.bin_conf.apply(weights_init)
    net.Norm.apply(weights_init)
    #net.trans_layers.apply(weights_init)
    #net.latent_layers.apply(weights_init)
    #net.cls_1000.apply(weights_init)
    if args.version == 'RFB_E_vgg':
        net.reduce.apply(weights_init)
        net.up_reduce.apply(weights_init)

else:
    # load resume network
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    # net = torch.nn.DataParallel(net)

if args.cuda:
    net.cuda()
    cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = optim.RMSprop(net.parameters(), lr=args.lr,alpha = 0.9, eps=1e-08,
#                      momentum=args.momentum, weight_decay=args.weight_decay)

# criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)
criterion = MultiBoxLoss_tf_source(num_classes, 0.5, True, 0, True, 3, 0.5, False)
# criterion = MultiBoxLoss_tf_source_combine(num_classes, 0.5, True, 0, True, 3, 0.5, False)

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()

def train():
    net.train()
    vis = Visualizer(env=Visdom['env'])
    trainer = MyTrainer()
    if args.cuda:
        trainer = trainer.cuda()
    # meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss #used to plot loss curve
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    if args.dataset == 'VOC':
        dataset = VOCDetection(VOCroot, train_sets, preproc(
            img_dim, rgb_means, p), AnnotationTransform())
    elif args.dataset == 'COCO':
        dataset = COCODetection(COCOroot, train_sets, preproc(
            img_dim, rgb_means, p))
    else:
        print('Only VOC and COCO are supported now!')
        return

    epoch_size = len(dataset) // args.batch_size
    max_iter = args.max_epoch * epoch_size

    #stepvalues_VOC = (150 * epoch_size, 200 * epoch_size, 250 * epoch_size)
    stepvalues_VOC = (90 * epoch_size, 120 * epoch_size, 140 * epoch_size)
    stepvalues_COCO = (90 * epoch_size, 120 * epoch_size, 140 * epoch_size)
    stepvalues = (stepvalues_VOC, stepvalues_COCO)[args.dataset == 'COCO']
    print('Training', args.version, 'on', dataset.name)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    lr = args.lr
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            epoch_t0 = time.time()    
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size,
                                                  shuffle=True, num_workers=args.num_workers,
                                                  collate_fn=detection_collate))
            loc_loss = 0
            conf_loss = 0
            binary_loss = 0

            trainer.reset_meters()
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                torch.save(net.state_dict(), args.save_folder + args.version + '_' + args.dataset + '_epoches_' +
                           repr(epoch) + '.pth')

            #calculate loss and mAP，show training curve in visdom
            

            epoch += 1
        

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        # print(batch_iterator)
        # print(next(batch_iterator))
        images, targets, img_ids = next(batch_iterator)

        # print('targets: ', targets)

        # print(np.sum([torch.sum(anno[:,-1] == 2) for anno in targets]))

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda()) for anno in targets]

        else:
            images = Variable(images)
            targets = [Variable(anno) for anno in targets]

        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_bin, _, _, _ = criterion(out, priors, targets)
        # loss = loss_l + loss_c + 0.1686 * loss_bin
        # loss_l, loss_c, _, _ = criterion(out, priors, targets)
        loss = loss_l + loss_c + loss_bin
        loss.backward()
        optimizer.step()

        # print('LossTuple: ', LossTuple(loss_l.item(), loss_c.item(), loss_bin.item(), loss.item()))
        trainer.update_meters(LossTuple(loss_l, loss_c, loss_bin, loss)) #calculate and save average losses

        if math.isinf(loss_l.item()):
            print('------------stop----------')
            print(loss_l)
            print('targets: ', targets)
            print('img_ids： ', img_ids)
            # return

        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        binary_loss += loss_bin.item()
        load_t1 = time.time()
        if (iteration + 1) % Visdom['plot_every'] == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' +
                  repr(iteration) + ' || L: %.4f C: %.4f B: %.4f||' % (
                      loss_l.item(), loss_c.item(), loss_bin.item()) +
                  'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))

            # show training curve in visdom // in single epoch, we draw average loss of all calculated iteration loss
            # print('LossTuple: ', LossTuple(loss_l, loss_c, loss_bin, loss))
            # print("trainer.get_meter_data(): ")
            print(trainer.get_meter_data())
            vis.plot_many(trainer.get_meter_data())

        #end of one epoch
        if (iteration + 1) % epoch_size == 0:            
            #print epoch loss and (TODO)mAP 
            epoch_t1 = time.time()  
            print('-------------------------')      
            #these loss values can be replaced by trainer.meters     
            print('Epoch:' + repr(epoch) + '/ ' + repr(epoch_size) + ' || L: %.4f C: %.4f B: %.4f||' % (
                      loc_loss/(iteration % epoch_size), conf_loss/(iteration % epoch_size), 
                      binary_loss/(iteration % epoch_size)) +
                  'Epoch time: %.4f min. ||' % ((epoch_t1 - epoch_t0)/60) + 'LR: %.8f' % (lr))
            print('-------------------------') 


    torch.save(net.state_dict(), args.save_folder +
               'Final_' + args.version + '_' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 6:
        lr = 1e-6 + (args.lr - 1e-6) * iteration / (epoch_size * 5)
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
