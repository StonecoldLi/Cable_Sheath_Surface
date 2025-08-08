import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob

# Import necessary functions from data_loader
from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP

# ------- 1. define loss function --------
bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(), loss5.data.item(), loss6.data.item()))
    return loss0, loss

# ------- 2. set the directory of training dataset --------

model_name = 'u2net'  # 'u2netp'

# Update paths to your dataset
data_dir = os.getcwd()  # Assuming the script is in the root directory
image_dir = './dataset/images'  # Path to images
label_dir = './dataset/masks'  # Path to masks

image_ext = '.png'  # Assuming the images are in jpg format
label_ext = '.png'  # Assuming the labels are in png format

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

epoch_num = 10000
batch_size_train = 12
batch_size_val = 1

# Collect all image paths
image_paths = glob.glob(os.path.join(data_dir, image_dir, '*' + image_ext))

# Collect corresponding label paths
label_paths = []
for img_path in image_paths:
    img_name = os.path.basename(img_path)  # Extract filename (without path)
    label_name = os.path.splitext(img_name)[0] + label_ext  # Replace extension with .png
    label_paths.append(os.path.join(data_dir, label_dir, label_name))

print("---")
print("train images: ", len(image_paths))
print("train labels: ", len(label_paths))
print("---")

# Number of training samples
train_num = len(image_paths)

# Prepare dataset and dataloader
salobj_dataset = SalObjDataset(
    img_name_list=image_paths,
    lbl_name_list=label_paths,
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)]))

salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

# ------- 3. define model --------
# Define the network based on the model name
if model_name == 'u2net':
    net = U2NET(3, 1)
elif model_name == 'u2netp':
    net = U2NETP(3, 1)

if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 2000  # Save the model every 2000 iterations

# Training loop
for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num += 1
        ite_num4val += 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # Wrap inputs and labels into Variables
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(), requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.data.item()
        running_tar_loss += loss2.data.item()

        # Clear temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

        # Save the model periodically
        if ite_num % save_frq == 0:
            torch.save(net.state_dict(), os.path.join(model_dir, model_name + "_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val)))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # Resume training
            ite_num4val = 0
