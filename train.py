import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader,random_split
from dataset.CamVid import CamVid
from dataset.voc import voc_dataset
from dataset.transform import train_transform
import os
from model.build_BiSeNet import BiSeNet
import torch
from tensorboardX import SummaryWriter
import tqdm
import numpy as np
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, \
    per_class_iu
from loss import DiceLoss
import time
from torch.utils.data import random_split
import random

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
# torch.backends.cudnn.deterministic = True

def val(args, model, dataloader):
    print('start val!')
    # label_info = get_label_info(csv_path)
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        start_time = time.time()    
        for i, (data, label) in enumerate(dataloader):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            # get RGB predict image
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label.cpu())

            # compute per pixel accuracy

            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)
        end_time = time.time()
        total_time = end_time - start_time  # 计算总时间
        total_frames = len(dataloader)  # 计算总帧数
        fps = total_frames / total_time
        precision = np.mean(precision_record)
        # miou = np.mean(per_class_iu(hist))
        miou_list = per_class_iu(hist)[:-1]
        # miou_dict, miou = cal_miou(miou_list, csv_path)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print('fps: %.3f' % fps)
        return precision, miou


def train(args, model, optimizer, dataloader_train, dataloader_val):
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))
    if args.loss == 'dice':
        loss_func = DiceLoss()
    elif args.loss == 'crossentropy':
        loss_func = torch.nn.CrossEntropyLoss()
    max_miou = 0
    step = 0
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i, (data, label) in enumerate(dataloader_train):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            output, output_sup1, output_sup2 = model(data)
            if args.dataset=='voc':
                output = output.squeeze(dim=1)
                output_sup1 = output_sup1.squeeze(dim=1)
                output_sup2 = output_sup2.squeeze(dim=1)
            loss1 = loss_func(output, label)
            loss2 = loss_func(output_sup1, label)
            loss3 = loss_func(output_sup2, label)
            loss = loss1 + loss2 + loss3
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_model_path, 'latest_dice_loss.pth'))

        if epoch % args.validation_step == 0:
            precision, miou = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                torch.save(model.module.state_dict(),
                           os.path.join(args.save_model_path, 'best_dice_loss.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=5, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=1, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')
    parser.add_argument('--model', type=str, default='BiSeNet-Pro', help='BiSeNet-Pro or BiSeNet-plus')

    args = parser.parse_args(params)

    # create dataset and dataloader
    if args.dataset == 'CamVid':
        train_path = os.path.join(args.data, 'train')
        train_label_path = os.path.join(args.data, 'train_labels')
        
        test_path = os.path.join(args.data, 'test')
        test_label_path = os.path.join(args.data, 'test_labels')
        csv_path = os.path.join(args.data, 'class_dict.csv')
        dataset_train = CamVid(train_path, train_label_path, csv_path, scale=(args.crop_height, args.crop_width),
                            loss=args.loss, mode='train')
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True
        )
        dataset_val = CamVid(test_path, test_label_path, csv_path, scale=(args.crop_height, args.crop_width),
                            loss=args.loss, mode='test')
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=1,
            shuffle=True,
            num_workers=args.num_workers
        )
        model = BiSeNet(args.num_classes, args.context_path, args.model)
    else:
        print("num_workers ",args.num_workers)
        crop = crop=(256,256)
        ratio = 0.9
        dataset=voc_dataset(root=args.data,transfrom=train_transform,crop_size=crop)
        train_size=int(len(dataset)*ratio)
        train_data,val_data=random_split(dataset,[train_size,len(dataset)-train_size])
        dataloader_train=DataLoader(train_data,batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers, drop_last=True)
        dataloader_val=DataLoader(val_data,batch_size=1,shuffle=True, num_workers=args.num_workers)
        model = BiSeNet(1, args.context_path, args.model)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')

    # train
    train(args, model, optimizer, dataloader_train, dataloader_val)

    # val(args, model, dataloader_val, csv_path)

if __name__ == '__main__':
    params = [
        '--num_epochs', '300',
        '--learning_rate', '2.5e-2',
        '--data', './CamVid',
        '--num_classes', '12',
        '--cuda', '0',
        '--batch_size', '4',  # 6 for resnet101, 12 for resnet18
        '--save_model_path', './checkpoints_18_sgd_camvid',
        '--context_path', 'resnet18', 
        '--optimizer', 'sgd',
        '--num_workers', '1',
        '--model', 'BiSeNet-plus'
    ]
    # params = [
    #     '--num_epochs', '300',
    #     '--dataset', 'voc',
    #     '--learning_rate', '2.5e-2',
    #     '--data', './voc2012',
    #     '--num_classes', '21',
    #     '--cuda', '0',
    #     '--batch_size', '4',  # 6 for resnet101, 12 for resnet18
    #     '--save_model_path', './checkpoints_18_sgd_voc',
    #     '--context_path', 'resnet18',  
    #     '--optimizer', 'sgd',
    #     '--num_workers', '1',
    #     '--model', 'BiSeNet-Pro'
    # ]
    main(params)

