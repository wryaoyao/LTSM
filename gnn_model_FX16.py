import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, JumpingKnowledge

def pairwise_concatenate_and_score(tensors, mlp):
    num_tensors, tensor_size = tensors.size()
    expanded1 = tensors.unsqueeze(1).expand(-1, num_tensors, -1)
    expanded2 = tensors.unsqueeze(0).expand(num_tensors, -1, -1)
    concatenated_matrix = torch.cat((expanded1, expanded2), dim=2)
    mask = torch.eye(num_tensors, dtype=torch.bool, device=tensors.device)
    concatenated_matrix = concatenated_matrix[~mask].view(-1, 2 * tensor_size)
    attention_scores = mlp(concatenated_matrix)
    return concatenated_matrix, attention_scores

class GIN_Net2(torch.nn.Module):
    def __init__(self, in_len=2000, in_feature=13, gin_in_feature=256, num_layers=1,
                 hidden=512, use_jk=False, pool_size=3, cnn_hidden=1, train_eps=True,
                 feature_fusion=None, class_num=7, esm_feat_dim=1280):
        super(GIN_Net2, self).__init__()
        self.use_jk = use_jk
        self.train_eps = train_eps
        self.feature_fusion = feature_fusion

        self.conv1d = nn.Conv1d(in_channels=in_feature, out_channels=cnn_hidden, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm1d(cnn_hidden)
        self.biGRU = nn.GRU(cnn_hidden, cnn_hidden, bidirectional=True, batch_first=True, num_layers=1)
        self.maxpool1d = nn.MaxPool1d(pool_size, stride=pool_size)
        self.global_avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.global_avgpool1d2 = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(math.floor(in_len / pool_size), gin_in_feature)
        
        # 即使esm_tensor为None，也要初始化这个层，以避免加载state_dict时出错
        self.esm_fc = nn.Linear(esm_feat_dim, gin_in_feature)
        
        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(gin_in_feature, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )

        self.gin_convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.gin_convs.append(GINConv(nn.Sequential(...))) # 保持结构一致
            
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.fc21 = nn.Linear(hidden, 64)
        self.fc22 = nn.Linear(hidden, 64)
        self.fc23 = nn.Linear(hidden, 64)
        self.fc24 = nn.Linear(hidden, 64)
        self.fc25 = nn.Linear(hidden, 64)
        self.fc26 = nn.Linear(hidden, 64)
        self.fc27 = nn.Linear(hidden, 64)
        self.fc3 = nn.Linear(64*7, class_num)

        self.ln1 = nn.Linear(gin_in_feature, gin_in_feature)
        self.ln2 = nn.Linear(gin_in_feature, gin_in_feature)
        self.ln3 = nn.Linear(gin_in_feature, gin_in_feature)
        self.ln4 = nn.Linear(gin_in_feature, gin_in_feature)
        self.ln5 = nn.Linear(gin_in_feature, gin_in_feature)
        self.ln6 = nn.Linear(gin_in_feature, gin_in_feature)
        self.ln7 = nn.Linear(gin_in_feature, gin_in_feature)

        
        self.fc_att_base = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.sigmoid_activation = nn.Sigmoid()
        
    
    
    def forward(self, x, edge_index, train_edge_id, p_matrix, fx, p=0.5, esm_tensor=None, tau_pred=1.0):
        
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = self.bn1(x)
        x = self.maxpool1d(x)
        x = x.transpose(1, 2)
        x, _ = self.biGRU(x)
        x = self.global_avgpool1d2(x)
        x = x.squeeze()
        x = self.fc1(x)
        
        if esm_tensor is not None:
             esm_embed = self.esm_fc(esm_tensor)
             x = esm_embed

        
        x1 = self.ln1(x)
        x2 = self.ln2(x)
        x3 = self.ln3(x)
        x4 = self.ln4(x)
        x5 = self.ln5(x)
        x6 = self.ln6(x)
        x7 = self.ln7(x)

        x1 = self.gin_conv1(x1, edge_index)
        x2 = self.gin_conv1(x2, edge_index)
        x3 = self.gin_conv1(x3, edge_index)
        x4 = self.gin_conv1(x4, edge_index)
        x5 = self.gin_conv1(x5, edge_index)
        x6 = self.gin_conv1(x6, edge_index)
        x7 = self.gin_conv1(x7, edge_index)

        f = None
        if fx:
            f1 = self.global_avgpool1d(x1[fx[0]].T).T
            f2 = self.global_avgpool1d(x2[fx[1]].T).T
            f3 = self.global_avgpool1d(x3[fx[2]].T).T
            f4 = self.global_avgpool1d(x4[fx[3]].T).T
            f5 = self.global_avgpool1d(x5[fx[4]].T).T
            f6 = self.global_avgpool1d(x6[fx[5]].T).T
            f7 = self.global_avgpool1d(x7[fx[6]].T).T
            f_stack = torch.stack([f1, f2, f3, f4, f5, f6, f7], dim=0)
            f_stack = torch.squeeze(f_stack)

            _, concatenated_matrix = pairwise_concatenate_and_score(f_stack, lambda t: t) # 只用它来做拼接

            att_logits = self.fc_att_base(concatenated_matrix)
            # 应用温度缩放
            att_logits_scaled = att_logits / tau_pred
            # 应用激活函数得到最终分数
            att = self.sigmoid_activation(att_logits_scaled)
            
            att = att.reshape(7, 6)

            f = torch.eye(7, device=x.device)
            fill_index = 0
            for i in range(7):
                for j in range(7):
                    if i != j:
                        f[i, j] = att.flatten()[fill_index]
                        fill_index += 1

        node_id = edge_index[:, train_edge_id]
        
        x1_out = F.relu(self.lin1(x1))
        x1_out = F.dropout(x1_out, p=p, training=self.training)
        x1_out = self.lin2(x1_out)
        x1_out = torch.mul(x1_out[node_id[0]], x1_out[node_id[1]])
        x1_out = self.fc21(x1_out)

        x2_out = F.relu(self.lin1(x2)); x2_out = F.dropout(x2_out, p=p, training=self.training); x2_out = self.lin2(x2_out); x2_out = torch.mul(x2_out[node_id[0]], x2_out[node_id[1]]); x2_out = self.fc22(x2_out)
        x3_out = F.relu(self.lin1(x3)); x3_out = F.dropout(x3_out, p=p, training=self.training); x3_out = self.lin2(x3_out); x3_out = torch.mul(x3_out[node_id[0]], x3_out[node_id[1]]); x3_out = self.fc23(x3_out)
        x4_out = F.relu(self.lin1(x4)); x4_out = F.dropout(x4_out, p=p, training=self.training); x4_out = self.lin2(x4_out); x4_out = torch.mul(x4_out[node_id[0]], x4_out[node_id[1]]); x4_out = self.fc24(x4_out)
        x5_out = F.relu(self.lin1(x5)); x5_out = F.dropout(x5_out, p=p, training=self.training); x5_out = self.lin2(x5_out); x5_out = torch.mul(x5_out[node_id[0]], x5_out[node_id[1]]); x5_out = self.fc25(x5_out)
        x6_out = F.relu(self.lin1(x6)); x6_out = F.dropout(x6_out, p=p, training=self.training); x6_out = self.lin2(x6_out); x6_out = torch.mul(x6_out[node_id[0]], x6_out[node_id[1]]); x6_out = self.fc26(x6_out)
        x7_out = F.relu(self.lin1(x7)); x7_out = F.dropout(x7_out, p=p, training=self.training); x7_out = self.lin2(x7_out); x7_out = torch.mul(x7_out[node_id[0]], x7_out[node_id[1]]); x7_out = self.fc27(x7_out)

        x_final = torch.cat([x1_out, x2_out, x3_out, x4_out, x5_out, x6_out, x7_out], dim=-1)
        x_final = self.fc3(x_final)


        return x_final, f

