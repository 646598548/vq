import torch
from cachesim import cache_sim_set_associative_fifo,cache_sim_fifo,cache_sim
dev=torch.device('cpu')
# cache = cache_sim(mode=None, linesize=4, cachesize=None, num_lin=256, codebook=code)

def create_index(index,dev,centroids):
    centroids=centroids.weight.view(1,65536,8)
    new_index = torch.zeros([index.shape[0], 2*index.shape[1]], dtype=torch.int32, device=dev)
    # print(index.shape)
    lower = index & 0xFFFF              # 提取低16位，形状 (448, 1792)
    upper = (index >> 16) & 0xFFFF      # 提取高16位，形状 (448, 1792)
    # print(lower.shape,upper.shape)s
    new_index = torch.stack([lower, upper], dim=1).view(index.shape[0], 2*index.shape[1])  

    return new_index

def recon(centroids,new_index):
    centroids = centroids.weight.view(65536, 8).to(dev)  # 修正拼写错误并准备张量
    recon_weight = torch.zeros([new_index.shape[0], new_index.shape[1]], device=dev)
    assert new_index.min() >= 0 and new_index.max() < 65536, "new_index 包含越界索引"
# 使用矢量化操作替代嵌套循环
    recon_weight = centroids[new_index, 0]
    print(recon_weight.shape)
    return recon_weight

def run_simulation(code,recon_weight):
    # 将输入数据转换为向量化操作
    cache=cache_sim_fifo(mode=None, linesize=4, cachesize=None, num_lin=256, codebook=code)
    recon_flatten = recon_weight.t().reshape(-1) #按行解码
    hit_cnt = torch.sum(torch.tensor([
        cache.sim(val) for val in recon_flatten
    ], dtype=torch.int32))
    hit_cnt=hit_cnt.item()
    sum_cnt=len(recon_flatten)
    print(f"Hit Rate: {hit_cnt / sum_cnt:.4f}")
    return None

for i in range(28):
    for k in ['mlp', 'self_attn']:  # 修正这里
        if k == 'mlp':
            for j in ['up_proj', 'down_proj', 'gate_proj']:
                indices_name = f'/home/jzgrp/xiaomenghan/vptq_qwen2-7/model.layers.{i}.{k}.{j}_indices.pt'
                centroids_name = f'/home/jzgrp/xiaomenghan/vptq_qwen2-7/model.layers.{i}.{k}.{j}_centroids.pt'
                
                centroids=torch.load(centroids_name).to(dev)
                index=torch.load(indices_name).to(dev)
                index=index.squeeze(0)
               
                new_index=create_index(index=index,dev=dev,centroids=centroids)
                recon_weight=recon(centroids=centroids,new_index=new_index)

                centroids = centroids.weight.view(16384, 32).to(device=dev, dtype=torch.float32)
                # 创建 code 张量，确保在 dev 上并为 float32
                code = torch.zeros([16384, 4], dtype=torch.float32, device=dev)
                code = centroids[:, [0, 8,16,24]]
                run_simulation(code=code,recon_weight=recon_weight)
                print(indices_name)
        else:
            for j in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                indices_name = f'/home/jzgrp/xiaomenghan/vptq_qwen2-7/model.layers.{i}.{k}.{j}_indices.pt'
                centroids_name = f'/home/jzgrp/xiaomenghan/vptq_qwen2-7/model.layers.{i}.{k}.{j}_centroids.pt'

                centroids=torch.load(centroids_name).to(dev)
                index=torch.load(indices_name).to(dev)
                index=index.squeeze(0)

                new_index=create_index(index=index,dev=dev,centroids=centroids)
                recon_weight=recon(centroids=centroids,new_index=new_index)

                centroids = centroids.weight.view(16384, 32).to(device=dev, dtype=torch.float32)
                # 创建 code 张量，确保在 dev 上并为 float32
                code = torch.zeros([16384, 4], dtype=torch.float32, device=dev)
                code = centroids[:, [0, 8,16,24]]

                run_simulation(code=code,recon_weight=recon_weight)
                print(indices_name)
