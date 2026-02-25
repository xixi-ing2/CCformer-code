import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self, nvars, dmodel, dff, drop=0.1):
        super(FeedForwardNetwork, self).__init__()
        
        # 第一个卷积层，输入和输出维度由 nvars 和 dmodel、dff 参数决定
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, 
                                 kernel_size=1, stride=1, padding=0, dilation=1, groups=nvars)
        
        # 激活函数
        self.ffn1act = nn.GELU()
        
        # 第二个卷积层，回到原始维度
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, 
                                 kernel_size=1, stride=1, padding=0, dilation=1, groups=nvars)
        
        # Dropout 层
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.ffn1drop1(self.ffn1pw1(x))  # Conv1d + Dropout
        x = self.ffn1act(x)                  # 激活函数 GELU
        x = self.ffn1drop2(self.ffn1pw2(x))  # Conv1d + Dropout
        return x
# nvars = 4     # 变量数，随意设置
# dmodel = 8    # 输入通道数
# dff = 16      # 中间层通道数
# drop = 0.1    # Dropout 概率

# # 实例化模块
# ffn = FeedForwardNetwork(nvars=nvars, dmodel=dmodel, dff=dff, drop=drop)

# # 创建输入张量，形状 [batch_size, channels, sequence_length]
# batch_size = 2
# sequence_length = 10
# x = torch.randn(batch_size, nvars * dmodel, sequence_length)  # 示例输入张量

# # 前向传播
# output = ffn(x)

# # 输出形状和结果检查
# print("Input shape:", x.shape)
# print("Output shape:", output.shape)