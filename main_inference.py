import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from datetime import datetime
from ast import literal_eval
from torchvision.utils import save_image
from torch.distributions.normal import Normal
from torch.distributions.log_normal import LogNormal
import copy
import numpy as np


def str2bool(v):
    v=v.strip()
    if isinstance(v, bool):
        print(v)
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='vgg_cifar10_inference',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
parser.add_argument('--a_bit', default=1, type=int, metavar='Acti',
                    help='activation precision')
parser.add_argument('--comp', default=False, type=str2bool, metavar='C',
                    help='Complementary or not')
parser.add_argument('--load_model', default='Linear_binary_new.pkl', type=str, metavar='MD',
                    help='Complementary or not')
parser.add_argument('--w_bit', default=1, type=int, metavar='W',
                    help='weight precision')      
parser.add_argument('--RH', default=1e7, type=float, metavar='RH',
                    help='RH')
parser.add_argument('--RL', default=1e4, type=float, metavar='RL',
                    help='RL')
parser.add_argument('--std', default=0.22, type=float, metavar='STD',
                    help='standard deviation')     
parser.add_argument('--sep', default=True, type=str2bool, metavar='S',
                    help='Weight separation or not')       
parser.add_argument('--finetune', default=False, type=str2bool, metavar='FT',
                    help='Fine tuning or not')                       
parser.add_argument('--Imax', default=300e-6, type=float, metavar='imax',
                    help='Maximum current for SA')  
parser.add_argument('--node', default=0.04, type=float, metavar='um',
                    help='technology node')  
parser.add_argument('--cell', default=50, type=float, metavar='U',
                    help='technology node') 
                    
def main():
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()

    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save == '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    model_config = {'input_size': args.input_size, 'dataset': args.dataset, 'a_bit': args.a_bit, 'comp':args.comp}

    if args.model_config != '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    logging.info("created model with configuration: %s", model_config)



    # create model
    temp  =     torch.load(args.load_model,map_location='cuda:0')
    state = temp['model']
    model.load_state_dict(state)    
    
    WLs = torch.ceil(torch.tensor(2*2*args.Imax/(1/args.RH + 1/args.RL)))   
    print("WLs =",WLs) 
    w_org = []
    num_SA = 0
    num_SA_power = 0
    num_driver = 0
    num_weight = 0
    cnt = 0
    layer_SA = []
    for name ,child in (model.named_children()):
        for x , y in (child.named_children()):
            if not ( isinstance(y, nn.BatchNorm2d) or isinstance(y, nn.BatchNorm1d)):
                for p in list(y.parameters()):

                    w_org.append(copy.deepcopy(p.data).cuda().mul(100).round())
                    
                    
                    if len(p.size()) == 4:
                        SA = torch.ceil(p.size()[3]*p.size()[2]*p.size()[1]/WLs)*p.size()[0]
                        layer_SA.append(SA)
                        num_SA = num_SA + SA


                        if args.comp == True:
                            num_driver = num_driver + SA + p.size()[3]*p.size()[2]*p.size()[1]*2*2
                        else:
                            num_driver = num_driver + SA + p.size()[3]*p.size()[2]*p.size()[1]*2
                        num_weight = num_weight + p.size()[3]*p.size()[2]*p.size()[1]*p.size()[0]
                    if len(p.size()) == 2 and p.size()[0]!=10:
                        SA = torch.ceil(p.size()[1]/WLs)*p.size()[0]
                        layer_SA.append(SA)
                        num_SA = num_SA + SA
                        if args.comp == True:
                            num_driver = num_driver + SA + p.size()[1]*2*2
                        else:
                            num_driver = num_driver + SA + p.size()[1]*2
                        num_weight = num_weight + p.size()[1]*p.size()[0]
                        
                cnt = cnt +1
    if args.sep == True:
        num_SA = num_SA*2
        layer_SA = [i*2 for i in layer_SA]
        num_weight = num_weight*2
        
    cell_area = torch.tensor(args.cell*args.node*args.node)
    Area_total = num_weight*cell_area + num_driver*cell_area*(256*256/(256*3)*0.94) + num_SA*cell_area*256
    print(num_weight)
    #memory 5.6e7
    
    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # Data loading code
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer,
                                           'lr': args.lr,
                                           'momentum': args.momentum,
                                           'weight_decay': args.weight_decay}})
    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    criterion.type(args.type)
    model.type(args.type)

    val_data = get_dataset(args.dataset, 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    train_data = get_dataset(args.dataset, 'train', transform['train'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

#     print(val_data[0][0].size(0))   #3*32*32
#     print(val_data[0][0].size(1))   #3*32*32
#     print(val_data[0][0].size(2))   #3*32*32

    # for i, (inputs, target) in enumerate(val_loader):
    #     print(inputs.size())
    #     print(target.size())
    model_list, weight_aomunt = model2list(model)
#     print(model_list)

    SA_num, layer_w, conv_count = model_flow(val_data[0][0], args.a_bit, layer_SA, model_list, p=False)

    SA_power = SA_num*0.6977*args.Imax
##################Set Resistance#########################
    RL = torch.tensor(args.RL); RL = RL.cuda(); GH = 1/RL;
    RH = torch.tensor(args.RH); RH = RH.cuda(); GL = 1/RH;
    G_dummy = (GH + GL)/2;
    # Gm = (GH + 2*GL)/3
    Seperation = args.sep
    fine_tune = args.finetune


    
    std = torch.tensor(args.std);
    

    
    GL_vari = LogNormal(torch.tensor([torch.log(GL)]), torch.tensor([std]))
    GH_vari = LogNormal(torch.tensor([torch.log(GH)]), torch.tensor([std]))
    # Gm_vari = LogNormal(torch.tensor([torch.log(Gm)]), torch.tensor([std]))

    bit = args.w_bit
    G_org = torch.linspace(-1,1,2**bit).mul(100).round().cuda()
    # G_device = torch.linspace((GH - GL),(GH - GL),2**bit)

    Gm_vari = []
    if Seperation:
        Gm = torch.linspace(1,-1,2**bit).cuda()*(GH - GL) + GL
        for i in range(2**(bit-1)):
            # print(i)
#             print(Gm[i])
            Gm_vari.append(LogNormal(torch.tensor([torch.log(Gm[i])]), torch.tensor([std])))
    else:
        Gm = torch.linspace(-1,1,2**bit).cuda()*(GH - G_dummy) + G_dummy
        for i in range(2**bit):
            # print(Gm[i])
            Gm_vari.append(LogNormal(torch.tensor([torch.log(Gm[i])]), torch.tensor([std])))

 



    if fine_tune:
        for param in model.parameters():
            param.requires_grad = False
            
            model.classifier[6].weight.requires_grad = True

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    logging.info('training regime: %s', regime)
    
    
    for epoch in range(args.start_epoch, args.epochs):
        array_p = 0
        cnt = 0
        if Seperation:
            
            scale = torch.tensor(1.0).cuda()/(GH - GL)
            
            for name ,child in (model.named_children()):
                for x , y in (child.named_children()):
                    if not ( isinstance(y, nn.BatchNorm2d) or isinstance(y, nn.BatchNorm1d)):
                        for p in list(y.parameters()):
                            GL_var = GL_vari.sample(p.size()).cuda().view(p.size())
                            
                            Gm_var= []
                            for i in range(2**(bit-1)):
                                Gm_var.append(Gm_vari[i].sample(p.size()).cuda().view(p.size()))
                            
                            linear_G = []
                            array_p_temp = 0
                            for i in range(2**(bit-1)):
                                linear_G.append((GL_var - Gm_var[i]).mul(scale))

                            for i in range(2**(bit-1)-1,-1,-1):
                                linear_G.append((Gm_var[i] - GL_var).mul(scale))

                            j = (2**bit-1)
                            for i in range(2**bit):
                                p.data[w_org[cnt] == G_org[i]] = linear_G[i][w_org[cnt] == G_org[i]];
                                if i<2**(bit-1):
                                    array_p_temp = array_p_temp + Gm_var[i][w_org[cnt] == G_org[i]].sum()
                                else:
                                    array_p_temp = array_p_temp + Gm_var[j][w_org[cnt] == G_org[i]].sum()
                                j = j - 1
                            array_p_temp = array_p_temp + GL*torch.numel(p.data)
                            if cnt<conv_count:
                                array_p_temp = array_p_temp*layer_w[cnt]*layer_w[cnt]
                                
                            array_p = array_p + array_p_temp


                            cnt += 1
                            
                            
        else:
            scale = torch.tensor(1.0).cuda()/(GH - G_dummy)

            for name ,child in (model.named_children()):
                for x , y in (child.named_children()):
                    if not ( isinstance(y, nn.BatchNorm2d) or isinstance(y, nn.BatchNorm1d)):
                        for p in list(y.parameters()):
                            
                            Gm_var= []
                            for i in range(2**bit):
                                Gm_var.append(Gm_vari[i].sample(p.size()).cuda().view(p.size()))

                            linear_G = []
                            for i in range(2**bit):             
                                linear_G.append((Gm_var[i] - G_dummy).mul(scale))
                            
                            for i in range(2**bit):
                                p.data[w_org[cnt] == G_org[i]] = linear_G[i][w_org[cnt] == G_org[i]];

                            cnt += 1
                            
        # evaluate on validation set
        # print(model.classifier[6].weight.data[0:6,0:6])
#         print(array_p/2)
#         print(SA_num*args.Imax*0.6977)
        power_total = array_p/2 + SA_num*args.Imax*0.6977 + array_p*0.6279/2
        # train for one epoch
        if fine_tune:
            train_loss, train_prec1, train_prec5 = train(
                train_loader, model, criterion, epoch, optimizer)
        
#         print(model.classifier[6].weight.data[0:6,0:6])

        
        val_loss, val_prec1, val_prec5 = validate(
            val_loader, model, criterion, epoch)
        print('Estimate total area = ',Area_total/1E6,' mm2')
        print('Estimate total energy = ',power_total*1E-8*10E3,' J')

        # print('Estimate total power = ',Power_total,' W')

        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'config': args.model_config,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'regime': regime
        }, is_best, path=save_path)
        logging.info('\n Epoch: {0}\t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(epoch + 1,  val_loss=val_loss,
                             val_prec1=val_prec1,
                             val_prec5=val_prec5))

        results.add(epoch=epoch + 1, val_loss=val_loss,
                    val_prec1 = val_prec1,
                    val_prec5 = val_prec5)

        results.save()


def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            target = target.cuda(non_blocking=True)
        input_var = Variable(inputs.type(args.type), volatile=not training)
        target_var = Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        
        if training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.data.copy_(p.org)
            optimizer.step()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data.clamp_(-1,1))
                    
                    


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)


if __name__ == '__main__':
    main()
