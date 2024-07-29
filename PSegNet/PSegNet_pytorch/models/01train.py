"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import provider
import model_pytorch as mp
import time
from tqdm import tqdm
from torchsummary import summary


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='model_pytorch', help='model name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=200, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.003, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-3, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--input_list_train', type=str, default='/data/train_file_list.txt',
                        help='Input data list file')
    parser.add_argument('--input_list_test', type=str, default='/data/test_file_list.txt',
                        help='Input data list file')

    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True
def main(args):

    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''Add summary writers'''
    train_writer = SummaryWriter(comment='train')
    test_writer = SummaryWriter(comment='test')
    NUM_CLASSES = 6 # sem classes
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    print("start loading training data ...")
    # Load train data,(3640,4096,3)
    train_file_list = provider.getDataFiles(os.path.join(ROOT_DIR, args.input_list_train))
    test_file_list = provider.getDataFiles(os.path.join(ROOT_DIR, args.input_list_test))
    TRAIN_DATASET = provider.PlantnetDataset(file_list=train_file_list)
    TEST_DATASET = provider.PlantnetDataset(file_list=test_file_list)
    print("start loading test data ...")
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=0, pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                                 pin_memory=True, drop_last=True)
    # weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.plantnet_model(NUM_CLASSES).to(device)
    criterion = MODEL.get_loss().cuda().to(device)
    classifier.apply(inplace_relu)
    summarytext = summary(classifier, (4096, 3))  # input_size为模型输入的数据大小，不包括batch_size
    print(summarytext)
    log_string(str(classifier))

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Conv1d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        # checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        checkpoint = torch.load(r'')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    #调整batchnorm层的momentum的值
    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-6 #1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0.0
    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, global_epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()
        for i, sample in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):#加载进度条
            optimizer.zero_grad()
            points = sample['data']
            group = sample['label']
            sem = sample['seg']
            points = points.data.numpy()
            group = group.data.numpy()
            sem = sem.data.numpy()
            pts_label_one_hot, pts_label_mask = mp.convert_seg_to_one_hot(sem)  # (16,4096,6)
            pts_group_label, pts_group_mask = mp.convert_groupandcate_to_one_hot(group)  # (16,4096,40)
            points = torch.Tensor(points)
            group = torch.Tensor(group)
            sem = torch.Tensor(sem)
            points, group, sem = points.float().cuda(), group.long().cuda(), sem.long().cuda()
            pred_sem, pred_ins, fuse_catch = classifier(points)
            loss, classify_loss, disc_loss, l_var, l_dist, l_reg, simmat_loss = criterion(pred_ins, group, pred_sem,
                                                                                          sem, fuse_catch,
                                                                                          pts_label_one_hot,
                                                                                          pts_group_label,
                                                                                          pts_group_mask)
            loss.backward()
            optimizer.step()
            pred_sem = pred_sem.argmax(dim=2)  # B X N
            pred_sem = pred_sem.cpu().detach().numpy()
            batch_label = sem.cpu().data.numpy()
            correct = np.sum(pred_sem == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
            if i % 50 == 0:
                log_string(
                    "loss: {:.2f}; sem_loss: {:.2f}; disc_loss: {:.2f}; l_var: {:.2f}; l_dist: {:.2f}; l_reg: {:.3f}; simmat_loss: {:.3f}.".format(
                        loss, classify_loss, disc_loss, l_var, l_dist, l_reg, simmat_loss))
            train_writer.add_scalar('Training loss', loss_sum / num_batches, global_epoch)
            train_writer.add_scalar('Training accuracy', total_correct / float(total_seen), global_epoch)
            train_writer.close()
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % 2 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model_epoch{}.pth'.format(epoch)
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            #labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()
            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))

            for i, sample in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = sample['data']
                group = sample['label']
                sem = sample['seg']
                points = points.data.numpy()
                group = group.data.numpy()
                sem = sem.data.numpy()
                pts_label_one_hot1, pts_label_mask1 = mp.convert_seg_to_one_hot(sem)  # (16,4096,6)
                pts_group_label1, pts_group_mask1 = mp.convert_groupandcate_to_one_hot(group)
                points = torch.Tensor(points)
                group = torch.Tensor(group)
                sem = torch.Tensor(sem)
                points, group, sem = points.float().cuda(), group.long().cuda(), sem.long().cuda()
                pred_sem, pred_ins, fuse_catch = classifier(points)
                loss, classify_loss, disc_loss, l_var, l_dist, l_reg, simmat_loss = criterion(pred_ins, group, pred_sem,
                                                                                              sem, fuse_catch,
                                                                                              pts_label_one_hot1,
                                                                                              pts_group_label1,
                                                                                              pts_group_mask1)
                loss_sum += loss
                pred_val = pred_sem.cpu().detach().numpy()
                pred_val = np.argmax(pred_val, 2)
                batch_label = sem.cpu().data.numpy()
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
                test_writer.add_scalar('loss/test', loss_sum / float(num_batches), global_step=global_epoch)
                test_writer.add_scalar('mIou/test', np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6)), global_step=global_epoch)
                test_writer.close()
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))
            log_string('------- IoU --------')
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))))
            log_string('eval point avg class IoU: %f' % (mIoU))

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Best mIoU: %f' % best_iou)

        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
