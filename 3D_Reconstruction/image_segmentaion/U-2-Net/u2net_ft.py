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
import csv  # 新增CSV支持
import datetime  # 新增时间戳
import pandas as pd

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

# ------- 2. Set directories and parameters --------
model_name = 'u2net'  # 必须与预训练模型类型一致
pretrained_path = './saved_models/u2net/u2net.pth'  # 预训练权重路径

# 数据集路径（根据实际情况修改）
data_dir = os.getcwd()
image_dir = './dataset/1/images'
label_dir = './dataset/1/masks'

flag = 1
# 输出目录配置
model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + str(flag) + '_finetune')
log_dir = os.path.join(model_dir, 'training_logs')
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# 训练参数
epoch_num = 100
batch_size_train = 12
save_frq = 100  # 每100次迭代保存一次模型

allowed_extensions = ['*.png', '*.jpg', '*.jpeg']  # 添加更多格式如 '*.bmp'

# 收集所有图像路径
image_paths = []
for ext in allowed_extensions:
    image_paths.extend(glob.glob(os.path.join(data_dir, image_dir, ext)))

# ------- 3. 数据加载 --------
# 获取数据路径
print(len(image_paths))
label_paths = glob.glob(os.path.join(data_dir, label_dir, '*.png'))
print(len(label_paths))

# 验证数据存在性
assert len(image_paths) == len(label_paths), "图像与标签数量不匹配"
print(f"\n训练图像: {len(image_paths)}张\n训练标签: {len(label_paths)}张\n")

# 创建数据集和数据加载器
salobj_dataset = SalObjDataset(
    img_name_list=image_paths,
    lbl_name_list=label_paths,
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)
    ])
)

salobj_dataloader = DataLoader(
    salobj_dataset,
    batch_size=batch_size_train,
    shuffle=True,
    num_workers=0
)

# ------- 4. 模型初始化 --------
# 创建模型
net = U2NET(3, 1) if model_name == 'u2net' else U2NETP(3, 1)

# 加载预训练权重
if os.path.exists(pretrained_path):
    state_dict = torch.load(pretrained_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    # 处理多GPU训练的权重
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict)
    print(f"成功加载预训练权重: {pretrained_path}")
else:
    raise FileNotFoundError(f"预训练模型未找到: {pretrained_path}")

# 冻结部分层（可选）
# for name, param in net.named_parameters():
#     if 'stage1' in name or 'stage2' in name:  # 示例：冻结前两个阶段
#         param.requires_grad = False

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# ------- 5. 优化器和日志配置 --------
# 仅优化需要梯度的参数
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, net.parameters()),
    lr=0.0001,  # 微调使用更小的学习率
    betas=(0.9, 0.999),
    weight_decay=0
)

# 创建CSV日志文件
csv_path = os.path.join(log_dir, f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'epoch', 'iteration', 'train_loss', 'target_loss'])

# ------- 6. 训练循环 --------
print("\n开始微调...")
total_iterations = 0
best_loss = float('inf')

for epoch in range(epoch_num):
    net.train()
    epoch_loss = 0.0
    epoch_target_loss = 0.0
    batch_count = 0

    for i, data in enumerate(salobj_dataloader):
        total_iterations += 1
        batch_count += 1

        # 数据准备
        inputs = data['image'].type(torch.FloatTensor).to(device)
        labels = data['label'].type(torch.FloatTensor).to(device)

        # 前向传播
        optimizer.zero_grad()
        d0, d1, d2, d3, d4, d5, d6 = net(inputs)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 记录损失
        epoch_loss += loss.item()
        epoch_target_loss += loss2.item()

        # 定期保存模型
        if total_iterations % save_frq == 0:
            torch.save(net.state_dict(), os.path.join(
                model_dir,
                f"{model_name}_finetune_iter{total_iterations}_loss{loss.item():.4f}.pth"
            ))

        # 清理内存
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

    # 记录epoch数据
    avg_epoch_loss = epoch_loss / batch_count
    avg_target_loss = epoch_target_loss / batch_count
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            epoch + 1,
            total_iterations,
            f"{avg_epoch_loss:.6f}",
            f"{avg_target_loss:.6f}"
        ])

    print(f"Epoch [{epoch+1}/{epoch_num}] "
          f"Train Loss: {avg_epoch_loss:.4f} "
          f"Target Loss: {avg_target_loss:.4f}")

# ------- 7. 最终保存和可视化 --------
# 保存最终模型
torch.save(net.state_dict(), os.path.join(model_dir, f"{model_name}_finetune_final.pth"))

# 绘制损失曲线
df = pd.read_csv(csv_path)
plt.figure(figsize=(12, 6))
plt.plot(df['epoch'], df['train_loss'], label='Training Loss')
plt.plot(df['epoch'], df['target_loss'], label='Target Loss')
plt.title("Fine-tuning Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(log_dir, "loss_curve.png"))
plt.show()

print("\n微调完成！模型和日志保存在:", model_dir)