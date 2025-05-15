import torch
from collections import Counter
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from cachesim import cache_sim_set_associative_fifo,cache_sim_fifo,cache_sim

dev=torch.device('cpu')
centroids=torch.load('/home/jzgrp/xiaomenghan/vptq_qwen2-7/model.layers.0.self_attn.q_proj_centroids.pt').to(dev)
index=torch.load('/home/jzgrp/xiaomenghan/vptq_qwen2-7/model.layers.0.self_attn.q_proj_indices.pt').to(dev)
print(index.shape)
centroids=centroids.weight.view(1,65536,8)
high_mask = 0xffff0000  # 未使用，可保留或删除
low_mask = 0x0000ffff   # 未使用，可保留或删除
new_index = torch.zeros([448, 3584], dtype=torch.int32, device=dev)
lower = index[0] & 0xFFFF              # 提取低16位，形状 (448, 1792)
upper = (index[0] >> 16) & 0xFFFF      # 提取高16位，形状 (448, 1792)
new_index = torch.stack([lower, upper], dim=2).view(448, 3584)  # 重组为 (448, 3584)
centroids = centroids.view(65536, 8).to(dev)  # 修正拼写错误并准备张量
print(centroids[52581])
recon_weight = torch.zeros([448, 3584], device=dev)
assert new_index.min() >= 0 and new_index.max() < 65536, "new_index 包含越界索引"
# 使用矢量化操作替代嵌套循环
recon_weight = centroids[new_index, 0]
centroids = centroids.view(16384, 32).to(device=dev, dtype=torch.float32)
# 创建 code 张量，确保在 dev 上并为 float32
code = torch.zeros([16384, 4], dtype=torch.float32, device=dev)
code = centroids[:, [0, 8, 16, 24]]

entry_character=torch.zeros([65536,448])
cindex=new_index.cpu()
for i in range(new_index.shape[0]):
    counts = torch.bincount(cindex[i, :], minlength=entry_character.shape[0])
    entry_character[:,i]=counts


data=entry_character.cpu().numpy()
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# best_k = int(input("根据图形输入最佳K值: "))  # 交互式选择
# 或者直接指定 best_k = 8

kmeans = MiniBatchKMeans(n_clusters=16,
                        batch_size=16384,
                        random_state=42)
cluster_labels = kmeans.fit_predict(data_scaled)

# 5. 保存结果
np.save('cluster_labels.npy', cluster_labels)
print("聚类完成，标签已保存为cluster_labels.npy")

# 可选：查看聚类中心
print("聚类中心形状:", kmeans.cluster_centers_.shape)

centroids = centroids.view(65536, 8).to(dev)  # 修正拼写错误并准备张量
sorted_indices = np.argsort(cluster_labels)
centroids = centroids[sorted_indices]
centroids = centroids.view(16384, 32).to(device=dev, dtype=torch.float32)
# 创建 code 张量，确保在 dev 上并为 float32
code = torch.zeros([16384, 4], dtype=torch.float32, device=dev)
code = centroids[:, [0, 8, 16, 24]]
print(code,sorted_indices)
print(cluster_labels[52581])

def run_simulation(code,recon_weight):
    # 将输入数据转换为向量化操作
    cache_l1=cache_sim(mode=None, linesize=4, cachesize=None, num_lin=256, codebook=code)
    cache_l2=cache_sim(mode=None, linesize=4, cachesize=None, num_lin=2048, codebook=code)
    recon_flatten = recon_weight.reshape(-1) #按行解码
    hit_cnt = torch.sum(torch.tensor([
        cache_l1.sim(val) for val in recon_flatten
    ], dtype=torch.int32))
    hit_cnt=hit_cnt.item()
    sum_cnt=len(recon_flatten)
    print(f"Hit Rate: {hit_cnt / sum_cnt:.6f}")
    hit_cnt = torch.sum(torch.tensor([
        cache_l2.sim(val) for val in recon_flatten
    ], dtype=torch.int32))
    hit_cnt=hit_cnt.item()
    sum_cnt=len(recon_flatten)
    print(f"Hit Rate: {hit_cnt / sum_cnt:.6f}")
    return None
run_simulation(code,recon_weight)