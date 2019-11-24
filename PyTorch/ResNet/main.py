"""
------------------------------------------------
File Name: main.py
Description: a ResNet example to classify dog and cat
Author: zhangtongxue
Date: 2019/10/24 11:19
-------------------------------------------------
"""
import argparse
import cv2
import time
import os
import torch
import torch.utils
import torchvision.datasets
import torch.utils.data
import numpy as np
import random
import warnings
import resnet

from PIL import Image, ImageDraw
from torch import nn, optim
from tqdm import tqdm
from torch.autograd import Variable
from torchvision import transforms
import torchvision.models as models

warnings.filterwarnings("ignore")


# 所有参数设置
class ModelConfig(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 是否GPU
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(48),
                transforms.CenterCrop(4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(48),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }
        self.data_dir = 'data/'  # 数据路径设置
        self.datasets_name = ['train', 'valid', 'test']  # 数据集存放路径

        self.resnet_model = resnet.resnet18(num_classes=2)  # 模型名字
        self.resume = False  # 重用模型
        self.pretrained_weight = 'best_resnet18.pth'
        self.model_save_dir = './ckpt/'  # 模型存放路径
        self.save_best_model_name = 'best_resnet18.pth'  # 历史最好的模型名字
        self.save_model_name = 'resnet18.pth'  # 模型保存名字

        self.batch_size = 2  # batch_size
        self.max_epoch = 3  # epoch
        self.lr = 1e-2  # 学习率
        self.lr_decay = 0.1  # 学习率衰减
        self.milestones = [20, 40]
        self.best_accuracy = -1  # 历史最优准确率
        self.start_epoch = 1

        self.test_image = 'data/test/cat/cat1.jpg'  # 测试图片


# 读取自定义数据:
# 方式一--每种类别图片放进单独文件夹
# https://blog.csdn.net/Summer_And_Opencv/article/details/88973713

# 还有方式二：继承torch.utils.data.Dataset类
# https://zhuanlan.zhihu.com/p/35698470
class MyDataSet(object):
    def __init__(self, model_config_object):
        self.transforms = model_config_object.data_transforms
        self.datasets_name = model_config_object.datasets_name
        self.data_dir = model_config_object.data_dir
        self.batch_size = model_config_object.batch_size

    # 读取数据
    def load_data(self):
        image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(self.data_dir, x),
                                                              self.transforms[x]) for x in self.datasets_name}

        data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                       batch_size=self.batch_size, num_workers=12, shuffle=True)
                        for x in self.datasets_name}

        data_size = {x: len(image_datasets[x]) for x in self.datasets_name}

        return image_datasets, data_loaders, data_size


# ResNet模型
class ModelCNN(object):
    def __init__(self, model_config_object):
        self.config = model_config_object
        self.model = self.config.resnet_model  # 建立模型
        self.model = self.model.to(self.config.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, self.config.milestones,
                                                        gamma=self.config.lr_decay,
                                                        last_epoch=-1)

    def train(self, ):
        # 是否加载已训练模型进行训练
        if self.config.resume:
            state = torch.load(os.path.join(self.config.model_save_dir, self.config.pretrained_weight),
                               map_location=self.config.device)
            self.model.load_state_dict(state['state_dict'])
            self.config.start_epoch = state['epoch']
            print('train with pretrained weight accuracy', state['accuracy'])

        # 读取数据集
        image_datasets, data_loaders, data_size = MyDataSet(self.config).load_data()
        train_loader, valid_loader, test_loader = data_loaders['train'], data_loaders['valid'], data_loaders['test'],

        train_size, valid_size, test_size = data_size['train'], data_size['valid'], data_size['test']
        print('train_size:%04d, valid_size:%04d, test_size:%04d\n' % (train_size, valid_size, test_size))

        for epoch in tqdm(range(self.config.start_epoch, self.config.max_epoch + 1)):

            loss_train, loss_valid, correct_train, correct_valid = 0, 0, 0, 0

            # 训练
            for batch_idx, (inputs, target) in enumerate(train_loader):
                inputs = inputs.to(self.config.device)
                target = target.to(self.config.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = (self.criterion(outputs, target)).sum()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_train += loss.item()
                correct_train += torch.sum(preds == target.data).to(torch.float32)

            # 验证
            with torch.no_grad():
                for batch_idx, (inputs, target) in enumerate(valid_loader):
                    inputs = inputs.to(self.config.device)
                    target = target.to(self.config.device)

                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = (self.criterion(outputs, target)).sum()

                    loss_valid += loss.item()
                    correct_valid += torch.sum(preds == target.data).to(torch.float32)

            # 训练和验证的精度计算
            train_accuracy = correct_train.data.cpu().numpy() / train_size
            valid_accuracy = correct_valid.data.cpu().numpy() / valid_size

            # 输出结果，保存模型状态
            state = {"state_dict": self.model.state_dict(), "epoch": epoch, "train_loss": loss_train,
                     "valid_loss": loss_valid, 'train_accuracy': train_accuracy, 'valid_accuracy': valid_accuracy,
                     'class_to_idx': image_datasets['test'].class_to_idx}
            torch.save(state,
                       os.path.join(self.config.model_save_dir, "epoch_%d_" % epoch + self.config.save_model_name))

            # 保存最佳模型状态
            if valid_accuracy > self.config.best_accuracy:
                self.config.best_accuracy = valid_accuracy
                torch.save(state, os.path.join(self.config.model_save_dir, self.config.save_best_model_name))
            print(
                'epoch:%04d, train loss:%.4f, valid loss:%.4f, train accuracy:%.4f, valid accuracy:%.4f, best accuracy:%.4f\n' % (
                    epoch, loss_train, loss_valid, train_accuracy, valid_accuracy, self.config.best_accuracy))

            # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
            self.scheduler.step()

    # 测试一张图片
    def inference(self, ):
        # 加载模型
        best_model = self.config.resnet_model
        state = torch.load(os.path.join(self.config.model_save_dir, self.config.save_best_model_name),
                           map_location=self.config.device)
        best_model.load_state_dict(state['state_dict'])

        best_model = best_model.to(self.config.device)
        best_model.eval()

        test_image = Image.open(self.config.test_image)
        test_image_transform = self.config.data_transforms['test'](test_image)

        # 输入模型测试
        outputs = self.model(test_image_transform.unsqueeze(0).to(self.config.device))
        _, preds = torch.max(outputs.data, 1)

        # 得到预测结果
        class_ = int(preds.data.cpu().numpy())
        class_to_idx = state['class_to_idx']
        category = list(class_to_idx.keys())[list(class_to_idx.values()).index(class_)]

        # 显示
        img = cv2.imread(self.config.test_image)
        cv2.imshow('result:' + category, img)
        cv2.waitKey()


if __name__ == "__main__":
    model_config_object = ModelConfig()
    ResNetModel = ModelCNN(model_config_object)
    # ResNetModel.train()
    ResNetModel.inference()