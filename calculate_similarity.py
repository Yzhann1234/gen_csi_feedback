import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import h5py

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数配置（与训练时保持一致）
parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--generator_path", type=str,
                    default=r'D:\PyTorch-GAN-master\PyTorch-GAN-master\implementations\gan1\generator.pth')
parser.add_argument("--data_path", type=str, default='CSI_featuresB.mat')
opt = parser.parse_args()


# 自定义数据集类（与训练时保持一致）
class CSIDataset(Dataset):
    def __init__(self, file_path):
        with h5py.File(file_path, 'r') as f:
            data = f['CSI_featuresA'][()]
        data = np.transpose(data, (3, 0, 1, 2))  # 形状 (114000, 2, 32, 12)
        max_val = data.max()
        min_val = data.min()
        self.data = 2 * ((data - min_val) / (max_val - min_val)) - 1
        self.data = torch.FloatTensor(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], 0


# 生成器定义（与训练时保持一致）
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.img_shape = (2, 32, 12)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


# 加载生成器
generator = Generator().to(device)
generator.load_state_dict(torch.load(opt.generator_path, map_location=device))
generator.eval()

# 加载真实数据集
dataset = CSIDataset(opt.data_path)

# 随机选择100个真实样本
real_indices = np.random.choice(len(dataset), 100000, replace=False)
real_samples = torch.stack([dataset[i][0] for i in real_indices]).to(device)

# 生成100个假样本
with torch.no_grad():
    z = torch.randn(100000, opt.latent_dim, device=device)
    fake_samples = generator(z)

# 计算平方广义余弦相似度
similarities = []
for fake, real in zip(fake_samples, real_samples):
    # 展平样本为向量
    fake_flat = fake.reshape(-1)
    real_flat = real.reshape(-1)

    # 计算余弦相似度并平方
    cos_sim = torch.cosine_similarity(fake_flat, real_flat, dim=0)
    squared_sim = cos_sim ** 2
    similarities.append(squared_sim.item())

# 计算并输出平均相似度
avg_similarity = np.mean(similarities)
print(f"Average Squared Generalized Cosine Similarity: {avg_similarity:.4f}")
