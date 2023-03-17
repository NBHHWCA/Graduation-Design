#用于在环境搭建和调试时 查看当前的torch和cuda是否安装成功 以及版本
import torch
print(torch.__version__)
print(torch.cuda.is_available())