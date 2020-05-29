from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from Project3_data import get_train_test_set, parse_line
from Project3_ResNet import resnet18, resnet34, resnet50, resnet101, resnet152
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import os
from collections import Counter
import torch.nn.functional as F

def main_test():
    # 1、进行参数设置
    parser = argparse.ArgumentParser(description="Detector_myself")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="input batch size for training(default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=64, metavar="N",
                        help="input batch size for testing(default: 64)")
    parser.add_argument("--epochs", type=int, default=200, metavar="N",
                        help="number of epochs to train(default: 100)")
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=117, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=80, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='learnt models are saving here')
    parser.add_argument('--phase', type=str, default='test',
                        # Train/train, Test/test
                        help='training, test')
    parser.add_argument('--checkpoint', type=str,
                        default='trained_models\\detector_epoch_199.pt',
                        help='run the specified checkpoint model')
    parser.add_argument('--retrain', action='store_true', default=False,
                        help='start training at checkpoint')
    parser.add_argument('--net', type=str, default='resnet101',
                        help='resnet 18, 34, 50, 101, 152')

    args = parser.parse_args()

    # 2、基本设置
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")  # cuda:0
    # 3、读取数据
    print('==> Loading Datasets')
    train_set, test_set = get_train_test_set()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size)
    # 4、将数据/网络传入CPU/GPU
    print('==> Building Model')

    # output categories
    categories = 62

    if args.net == 'ResNet18' or args.net == 'resnet18':
        model = resnet18()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, categories)
        model = model.to(device)
    elif args.net == 'ResNet34' or args.net == 'resnet34':
        model = resnet34()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, categories)
        model = model.to(device)
    elif args.net == 'ResNet50' or args.net == 'resnet50':
        model = resnet50()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, categories)
        model = model.to(device)
    elif args.net == 'ResNet101' or args.net == 'resnet101':
        model = resnet101()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, categories)
        model = model.to(device)
    elif args.net == 'ResNet152' or args.net == 'resnet152':
        model = resnet152()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, categories)
        model = model.to(device)

    #model = Net().to(device)
    if args.phase == 'Test' or args.phase == 'test':
        model_name = args.checkpoint
        model.load_state_dict(torch.load(model_name))
        model.eval()

    # 5、定义损失函数和优化器

    # parameter of weighted cross entropy
    def data_weight(phase):
        data_file = phase + '_label' + '.csv'
        data_file = os.path.join('traffic-sign', data_file)
        with open(data_file) as f:
            lines = f.readlines()[1:]

        cate_list = []
        for i in range(len(lines)):
            cate = lines[i].strip().split(',')
            cate_list.append(cate[2])
        #cate = [j[2] for j in [parse_line(i) for i in lines]]
        w = Counter(cate_list)
        total = sum(w.values())
        w_list = []
        for i in range (len(w)):
            x = w[str(i)]/total
            w_list.append(x)
        return w_list

    weights = data_weight(args.phase)
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    # 6、Scheduler Step
    # scheduler= optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
    # step_size(int) - 学习率下降间隔数,若为30,则会在30, 60, 90…个step时，将学习率调整为lr * gamma
    # gamma(float) - 学习率调整倍数，默认为0.1倍，即下降10倍
    # last_epoch(int) - 上一个epoch数，这个变量用来指示学习率是否需要调整。
    # 当last_epoch符合设定的间隔时，就会对学习率进行调整,当为 - 1时,学习率设置为初始值

    # 7、定义程序所处阶段
    if args.phase == "Train" or args.phase == "train":

        print('==> Start Training')
        if args.retrain:
            model.load_state_dict(torch.load(args.checkpoint))
            print("Training from checkpoint %s" % args.checkpoint)
        train_losses, valid_losses = train(args, train_loader, valid_loader,
                                           model, criterion, optimizer, device)
        print("Learning Rate:", args.lr, "Epoch:", args.epochs, "Seed:",
              args.seed, "Batch_Size:", args.batch_size, "Optimizer:", optimizer)
        print('====================================================')
        #loss_show(train_losses, valid_losses, args)

    elif args.phase == 'Test' or args.phase == "test":
        print('==> Testing')
        with torch.no_grad():
            result, valid_mean_pts_loss, accuracy= test(valid_loader, model, criterion, device)
            print(valid_mean_pts_loss)
            print(accuracy)
        # 利用预测关键点随机作出图像与真实值对比
        result_show(result)

def train(args, train_loader, valid_loader, model, criterion, optimizer, device):
    # 设定保存
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)
    # 设定训练次数、损失函数
    epoch = args.epochs
    # monitor training loss
    train_losses = []
    valid_losses = []
    log_path = os.path.join(args.save_directory, 'log_info.txt')
    if os.path.exists(log_path):
        os.remove(log_path)
    # 开始训练模型
    for epoch_id in range(epoch):
        # training the model
        model.train()
        log_lines = []

        #params for statistic
        train_pred_correct = 0
        # train_pred_correct_zero = 0
        # train_pred_correct_one = 0
        # train_zero_num = 0
        # train_one_num = 0
        valid_pred_correct = 0
        # valid_pred_correct_zero = 0
        # valid_pred_correct_one = 0
        # valid_zero_num = 0
        # valid_one_num = 0

        for batch_idx, batch in enumerate(train_loader):
            # input
            img = batch['image']
            input_img = img.to(device)
            #print(input_img.shape)
            # ground truth
            cls = batch['class']
            target_cls = cls.to(device)
            #print(target_cls)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            output_cls = model(input_img)
            #print(output_cls)
            #statistic
            _, pred_train = torch.max(output_cls.data, 1) #比较dim为1情况下的最大值，返回最大值和最大值对应下标
            # # 正样本为[0,1]，负样本为[1,0]
            train_pred_correct += (pred_train == target_cls).sum()
            # train_pred_correct_zero += ((pred_train==0)&(target_cls==0)).sum()
            # train_pred_correct_one += ((pred_train==1)&(target_cls==1)).sum()
            # train_zero_num += (target_cls==0).sum()
            # train_one_num += (target_cls==1).sum()

            # First: classification
            loss = criterion(output_cls, target_cls)

            # do BP automatically
            loss.backward()
            optimizer.step()

            # show log info
            if batch_idx % args.log_interval == 0:
                log_line = 'Train Epoch:{}[{}/{}({:.0f}%)]\t loss:{:.6f}'.format(
                    epoch_id,
                    batch_idx * len(img),  # 批次序号*一批次样本数=已测样本数
                    len(train_loader.dataset),  # 总train_set样本数量
                    100. * batch_idx / len(train_loader),  # 以上两者之比
                    loss.item()
                )
                print(log_line)

                log_lines.append(log_line)
        train_losses.append(loss)

        Train_CLS_accuracy = train_pred_correct.item() / len(train_loader.dataset)

        log_line_train_accuracy = 'Train_CLS_accuracy:{:.4f}% ({} / {})'.format(
            100 * Train_CLS_accuracy,train_pred_correct.item(),len(train_loader.dataset))
        print(log_line_train_accuracy)
        log_lines.append(log_line_train_accuracy)

        # 验证（使用测试数据集）
        valid_mean_pts_loss = 0.0
        model.eval()  # prep model for evaluation
        with torch.no_grad():
            valid_batch_cnt = 0
            for batch_idx, batch in enumerate(valid_loader):
                valid_batch_cnt += 1
                valid_img = batch['image']
                valid_img = valid_img.to(device)
                # ground truth
                valid_cls = batch['class']
                valid_target_cls = valid_cls.to(device)
                #print("valid_target_cls = ", valid_target_cls, valid_target_cls.shape)
                # result
                valid_output_cls = model(valid_img)

                #statistic
                _, pred_valid = torch.max(valid_output_cls.data, 1)
                valid_pred_correct += (pred_valid == valid_target_cls).sum()

                # Valid CLS Loss
                valid_loss= criterion(valid_output_cls, valid_target_cls)

                valid_mean_pts_loss += valid_loss.item()

            # 结论输出
            valid_mean_pts_loss /= valid_batch_cnt * 1.0

            log_line = 'Valid: loss: {:.6f}'.format(valid_mean_pts_loss)
            print(log_line)
            log_lines.append(log_line)
            valid_losses.append(valid_mean_pts_loss)

            Valid_CLS_accuracy = valid_pred_correct.item() / len(valid_loader.dataset)
            log_line_valid_accuracy = 'Valid_CLS_accuracy:{:.4f}% ({} / {})'.format(
                100 * Valid_CLS_accuracy,valid_pred_correct.item(),len(valid_loader.dataset))

            log_lines.append(log_line_valid_accuracy)
            print(log_line_valid_accuracy)
        print('=============================')
        # save model
        if args.save_model:
            saved_model_name = os.path.join(args.save_directory,
                                            'detector_epoch' + '_' + str(epoch_id) + '.pt')
            torch.save(model.state_dict(), saved_model_name)
        # write log info
        with open(log_path, "a") as f:
            for line in log_lines:
                f.write(line + '\n')
    return train_losses, valid_losses

# 测试，计算测试集上预测关键点
def test(valid_loader, model, criterion, device):
    valid_mean_pts_loss = 0.0
    valid_batch_cnt = 0
    result = []

    test_pred_correct = 0

    for batch_idx, batch in enumerate(valid_loader):
        valid_batch_cnt += 1
        valid_img = batch['image']
        input_img = valid_img.to(device)
        # ground truth
        cls = batch['class']
        target_cls = cls.to(device)
        # result
        output_cls = model(input_img)
        # loss_cls
        loss_cls = criterion(output_cls, target_cls)
        print(loss_cls)
        # # accuracy
        _, pred_test = torch.max(output_cls.data, 1)
        test_pred_correct += (pred_test == target_cls).sum()

        valid_mean_pts_loss += loss_cls.item()
        device2 = torch.device('cpu')
        output_cls = output_cls.to(device2)
        for i in range(len(valid_img)):
            sample = {
                'image': valid_img[i],
                'class': output_cls[i],
            }
            result.append(sample)
    # 计算loss值
    valid_mean_pts_loss /= valid_batch_cnt * 1.0
    # accuracy
    test_CLS_accuracy = test_pred_correct.item() / len(valid_loader.dataset)

    test_accuracy = 'Test_CLS_accuracy:{:.4f}% ({} / {})'.format(
        100 * test_CLS_accuracy,test_pred_correct.item() , len(valid_loader.dataset))
    print("Test_accuracy = ", test_accuracy)
    return result, valid_mean_pts_loss


if __name__ == '__main__':
    main_test()

