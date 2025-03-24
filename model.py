import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Bottlrneck(torch.nn.Module):
    def __init__(self,In_channel,Med_channel,Out_channel,downsample=False):
        super(Bottlrneck, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm1d(Out_channel),
            torch.nn.ReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv1d(In_channel, Out_channel,1,self.stride)
        else:
            self.res_layer = None

    def forward(self,x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x)+residual


class ResNet(torch.nn.Module):
    def __init__(self,in_channels=1,classes=4):
        super(ResNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels,64,kernel_size=7,stride=2,padding=3),
            torch.nn.MaxPool1d(3,2,1),

            Bottlrneck(64,64,256,False),
            Bottlrneck(256,64,256,False),
            Bottlrneck(256,64,256,False),
            
            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(512,1)
        )

    def forward(self,x):
        x = x.reshape(-1, 1, x_train.shape[1])
        x = self.features(x)
        x = x.view(-1,512)
        return x


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(MultiheadAttentionBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        att_output, _ = self.att(x, x, x)
        out1 = self.layernorm1(x + self.dropout1(att_output))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output))
        return out2

class MHA_ResNet (nn.Module):
    def __init__(self, in_channels=1, embed_dim=512, num_heads=8, ff_dim=2048, num_transformer_blocks=1, dropout=0.1):
        super(MHA_ResNet, self).__init__()
        self.resnet = ResNet(in_channels)
        self.embed_dim = embed_dim
        
        self.transformer_blocks = nn.ModuleList(
            [MultiheadAttentionBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_transformer_blocks)]
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x):
        x = x.reshape(-1, 1, x.shape[1])
        x = self.resnet.features(x)
        x = x.view(-1, self.embed_dim)
        x = x.unsqueeze(0) 
        
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        x = x.squeeze(0) 
        x = self.classifier(x)
        return x
