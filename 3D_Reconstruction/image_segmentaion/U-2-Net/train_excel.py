import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import glob
import matplotlib.pyplot as plt
import pandas as pd  # 导入 pandas 库

# Import necessary functions from data_loader
from data_loader import RescaleT, RandomCrop, ToTensorLab, SalObjDataset
from model import U2NET, U2NETP

# ------- 1. Define loss function --------
bce_loss = nn.BCELoss(reduction='mean')

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f" % (
        loss0.data.item(), loss1.data.item(), loss2.data.item(),
        loss3.data.item(), loss4.data.item(), loss5.data.item(), loss6.data.item()))

    return loss0, loss

# ------- 2. Set the directory of training dataset --------

model_name = 'u2net'  # 'u2netp'

# Update paths to your dataset
data_dir = os.getcwd()  # Assuming the script is in the root directory
image_dir = './dataset/images'  # Path to images
label_dir = './dataset/masks'  # Path to masks

image_ext = '.png'  # Assuming the images are in png format
label_ext = '.png'  # Assuming the labels are in png format

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir, exist_ok=True)

epoch_num = 300
batch_size_train = 15
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

salobj_dataloader = DataLoader(
    salobj_dataset,
    batch_size=batch_size_train,
    shuffle=True,
    num_workers=0  # 在 Windows 上将 num_workers 设置为 0
)

# ------- 3. Define model --------
# Define the network based on the model name
if model_name == 'u2net':
    net = U2NET(3, 1)
elif model_name == 'u2netp':
    net = U2NETP(3, 1)

if torch.cuda.is_available():
    net.cuda()

# ------- 4. Define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

if __name__ == '__main__':

    # ------- 5. Training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 1  # 将初始值设为 1，避免除以零
    save_frq = 100  # Save the model every 100 iterations

    # 用于记录每次迭代的损失
    loss_records = []

    # 用于记录每个 epoch 的训练损失和目标损失
    train_losses = []
    target_losses = []

    # Training loop
    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num += 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # Wrap inputs and labels into Variables
            if torch.cuda.is_available():
                inputs_v = Variable(inputs.cuda(), requires_grad=False)
                labels_v = Variable(labels.cuda(), requires_grad=False)
            else:
                inputs_v = Variable(inputs, requires_grad=False)
                labels_v = Variable(labels, requires_grad=False)

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

            # 计算平均损失，避免除以零
            train_loss = running_loss / ite_num4val if ite_num4val != 0 else 0.0
            tar_loss = running_tar_loss / ite_num4val if ite_num4val != 0 else 0.0

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %.6f, tar: %.6f" % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num,
                train_loss, tar_loss))

            # 记录每次迭代的损失
            loss_records.append({
                'iteration': ite_num,
                'epoch': epoch + 1,
                'batch': i + 1,
                'train_loss': train_loss,
                'tar_loss': tar_loss
            })

            ite_num4val += 1  # 增加计数

            # Record losses for plotting
            if ite_num4val % (train_num // batch_size_train) == 0:
                train_losses.append(train_loss)
                target_losses.append(tar_loss)
                running_loss = 0.0
                running_tar_loss = 0.0
                ite_num4val = 1  # 重置为 1，避免除以零

            # Save the model periodically
            if ite_num % save_frq == 0:
                torch.save(net.state_dict(), os.path.join(
                    model_dir,
                    model_name + "_bce_itr_%d_train_%.6f_tar_%.6f.pth" % (
                        ite_num, train_loss, tar_loss)))

    # 训练结束后保存损失到 Excel 文件
    df_losses = pd.DataFrame(loss_records)
    excel_output_path = os.path.join(model_dir, 'training_losses.xlsx')
    df_losses.to_excel(excel_output_path, index=False)

    # 训练结束后绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue', linestyle='-', linewidth=2)
    plt.plot(target_losses, label='Target Loss', color='red', linestyle='--', linewidth=2)
    plt.title("Training Loss and Target Loss Over Epochs", fontsize=16)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存训练损失图
    plt.savefig(os.path.join(model_dir, "training_loss_curve.png"))
    plt.show()
