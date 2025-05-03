# Load model directly

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import os,sys
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import math
os.environ["CUDA_VISIBLE_DEVICES"]="5"
# model_path="/data/xiaomenghan/Llama-3.2-1B"
model_path="/data/jiyoushu/hub/Qwen/Qwen2___5-Coder-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,  # 使用半精度加速推理
        trust_remote_code=True
).eval()

print(model.model.layers[0].self_attn.q_proj.weight)
# def find_layers(module, layers=[nn.Linear], name=''):
#     if type(module) in layers:
#         return {name: module}
#     res = {}
#     for old_name, child in module.named_children():
#         res.update(find_layers(child, layers=layers, name=name + '.' + old_name if name != '' else old_name))
#     return res
# # operators = find_layers(model.model.layers[0])
# # opeartor_names = [list(operators.keys())]
# #     # with torch.no_grad():
# # for names in opeartor_names:
# #         # subset: (op name, op) pairs
# #         subset = {n: operators[n] for n in names}
# #         print((subset['self_attn.q_proj'].weight))

# def reorder_weight (name,layer_idx,subset,iratio):
#         name2weight = {
#                     'self_attn.v_proj': 'qkv',
#                     'self_attn.q_proj': 'qkv',
#                     'self_attn.k_proj': 'qkv',
#                     'self_attn.o_proj': 'o',
#                     'mlp.up_proj': 'up',
#                     'mlp.gate_proj': 'up',
#                     'mlp.down_proj': 'down'
#                 }
#         address_name = f'{layer_idx}_{name2weight[name]}.pt'
#         index=torch.load(address_name)
#         processing_data=subset[name].weight.t()
#         im_index=math.floor(iratio*processing_data.shape[1])
#         unim_index=processing_data.shape[1]-im_index
#         reorder_weight=torch.index_select(processing_data,1,index)

#         unimportant_tensor=reorder_weight[:,0:unim_index-1].t()
#         important_tensor=reorder_weight[:,unim_index:im_index-1].t()

#         return unimportant_tensor,important_tensor





# qkv_modules = ['q_proj', 'k_proj', 'v_proj']
# up_modules = ['gate_proj' , 'up_proj']
# down_modules = ['down_proj']
# o_modules = ['o_proj']

# count=0
# hessian_name='1'
# # I=torch.zeros([2048])
# for name, module in model.named_modules():
        
#         if 'gate_proj' in name:
#                 if any(t in name for t in up_modules):
#                         for i in range(15,-1,-1):
#                                 if f'{i}' in name:
#                                         count=i
#                                         break
#                         hessian_name=f'/home/jzgrp/xiaomenghan/quip-sharp/hessians/llama32-1B/{count}_qkv.pt'
#                 # print(hessian_name)
#                 # weight_hessian=torch.load(hessian_name)
#                 # hessian=weight_hessian['flatH']

#         # if any(t in name for t in qkv_modules):
#         if 'gate_proj' in name:
#                 print(3)
#                 processing_data=module.weight.data.t().abs()
#                 I=torch.zeros([processing_data.shape[1]])
#                 mean=torch.mean(processing_data,dim=0)
#                 for j in range(processing_data.shape[1]):
#                         I[j]=mean[j]
#                 sorted_tensor, indices = torch.sort(I)
#                 torch.save(indices,f'./1b_weight_mean_index/{count}_gate_index.pt')






# qkv_modules = ['q_proj', 'k_proj', 'v_proj']
# for name, module in model.named_modules():
#         if 'q_proj' in name:
#                 if any(t in name for t in qkv_modules):
#                         for i in range(15,-1,-1):
#                                 if f'{i}' in name:
#                                         count=i
#                                         break
#                         index_name=f'/home/jzgrp/xiaomenghan/vq-test/q_weight_index/{count}_q_mean_weight.pt'
#                         index=torch.load(index_name).to(model.device)
#                         # print(index.device)
#         if 'q_proj' in name:
#         # if any(t in name for t in qkv_modules):
#                 processing_data=module.weight.data.t().abs()
#                 reorder_weight=torch.index_select(processing_data,1,index)
#                 # row_sum=reorder_weight.sum(dim=0)
#                 # print(name,row_sum,reorder_weight[0,:])
#                 # print(row_sum)
#                 # # print(reorder_weight.device)
#                 z=reorder_weight.abs().cpu().detach()
                # z=plot_tensor.abs().detach().cpu()
                # step_x, step_y = 1, 1
                # z_sampled = z[::step_x, ::step_y]
                # # 生成行和列索引
                # x_indices = np.arange(0, z.shape[0], step_x)
                # y_indices = np.arange(0, z.shape[1], step_y)
                # X, Y = np.meshgrid(x_indices, y_indices, indexing='ij')  # 确保网格与数据对齐

                # # 绘图
                # fig = plt.figure(figsize=(12, 8))
                # ax = fig.add_subplot(111, projection='3d')

                # # 调整rstride和cstride以加速渲染
                # ax.plot_surface(X, Y, z_sampled, cmap='viridis', rstride=5, cstride=5, 
                #                 linewidth=0, antialiased=False)

                # ax.set_xlabel('Row (X axis)')
                # ax.set_ylabel('Column (Y axis)')
                # ax.set_zlabel('Absolute Value')
                # plt.title('3D Surface Plot (Downsampled)')
                # plt.savefig(f"./qkv_weight_reorder/{count}_q_mean_weight.png")

                





                
                

                        

        
