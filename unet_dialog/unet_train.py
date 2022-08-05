# unet\unet_train.py

import datetime
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image
from sklearn.metrics import f1_score
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from unet_model import *


def display_train_img(x, y_true, y_pred):
    plt.clf()

    fig1 = torchvision.utils.make_grid(x.cpu(), normalize=True).permute(1, 2, 0)
    fig2 = torchvision.utils.make_grid(y_true.cpu(), normalize=True).permute(1, 2, 0)
    fig3 = torchvision.utils.make_grid(y_pred.cpu(), normalize=True).permute(1, 2, 0)

    fig = plt.figure("img")

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.imshow(fig1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.imshow(fig2)
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.imshow(fig3)

    plt.show(block=False)
    plt.draw()
    plt.pause(0.1)


class dataset(Dataset):
    def __init__(self, phase, args):
        if phase != 'train' and phase != 'test':
            raise Exception('phase : train or test')

        self.img_list = []
        self.label_list = []

        for ext in ('*.bmp', '*.png', '*.jpg', '*.tif'):
            self.img_list.extend(glob.glob(os.path.join(args.img_path, phase, ext)))
            self.label_list.extend(glob.glob(os.path.join(args.img_path, phase + '_label', ext)))

        if len(self.img_list) != len(self.label_list):
            raise Exception('not match %s img, label' % phase)

        if phase == 'train':
            self.transform = transforms.Compose([transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
                                                 transforms.RandomVerticalFlip(p=0.5),
                                                 transforms.RandomHorizontalFlip(p=0.5)])
        else:
            self.transform = transforms.Compose([])

        self.transform_img = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                                                 transforms.ToTensor()])

        self.transform_label = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                                                   transforms.ToTensor()])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        label = Image.open(self.label_list[idx]).convert('L')
        img.putalpha(label)
        img = self.transform(img)
        label = img.split()[-1]
        img = img.convert('RGB')

        return self.transform_img(img), \
               self.transform_label(label), \
               os.path.basename(self.img_list[idx])


def segmentation(args):
    model = UNet()
    model.to(args.device)

    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    dataloader_train = DataLoader(dataset=dataset('train', args),
                                  batch_size=args.batch_size,
                                  shuffle=True)

    dataloader_test = DataLoader(dataset=dataset('test', args),
                                 batch_size=1,
                                 shuffle=False)


    # train
    for epoch in range(args.num_epochs):
        model.train()
        torch.set_grad_enabled(True)
        sum_loss = 0

        for x, y_true, fname in dataloader_train:
            x, y_true = x.to(args.device), y_true.to(args.device)

            y_pred = model(x)

            loss = criterion(y_pred, y_true)
            sum_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # display_train_img(x, y_true, y_pred)

        print('%s  |  Epoch %2d / %2d  |  Loss %7.3f' %
              (datetime.datetime.now(), epoch + 1, args.num_epochs, sum_loss))



    save_dirname = 'result_%s' % args.model_name
    os.makedirs(os.path.join(save_dirname, 'img'), exist_ok=True)

    torch.save(model.state_dict(),
               os.path.join(save_dirname, 'unet_%s_epoch%06d.pth' % (args.model_name, args.num_epochs)))

    # test
    model.eval()
    torch.set_grad_enabled(False)
    for x, y_true, fname in dataloader_test:
        x, y_true = x.to(args.device), y_true.to(args.device)

        y_pred = model(x)

        x = (x.cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        y_true = (y_true.cpu().numpy()[0][0] * 255).astype(np.uint8)
        y_pred = (y_pred.cpu().numpy()[0][0] * 255).astype(np.uint8)
        y_pred = cv2.threshold(y_pred, 128, 255, cv2.THRESH_BINARY)[1]

        cv2.imwrite(os.path.join(save_dirname, 'img', fname[0]),
                    cv2.vconcat([cv2.cvtColor(x, cv2.COLOR_RGB2BGR),
                                 cv2.cvtColor(y_true, cv2.COLOR_GRAY2BGR),
                                 cv2.cvtColor(y_pred, cv2.COLOR_GRAY2BGR)]))

    os.startfile(os.path.join(save_dirname, 'img'))


class parameters:
    def __init__(self):
        self.model_name = 'fabric'
        self.img_path = r'fabric'
        # {img_path}\\train\\a.png
        # {img_path}\\train_label\\a.png
        # {img_path}\\test\\b.png
        # {img_path}\\test_label\\b.png
        self.num_epochs = 100
        self.batch_size = 20
        self.img_size = 128
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


args = parameters()
segmentation(args)
