import os
import time
import math
import random
import pickle
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F 
from gnn_data_FX16 import GNN_DATA
from gnn_model_FX16 import GIN_Net2
from utils import Metrictor_PPI, print_file
from BS_ES_NS import BS_ES_NS, BS_ES_NS_metrics
from tensorboardX import SummaryWriter


np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--description', default=None, type=str,
                    help='train description')
parser.add_argument('--ppi_path', default='./data/protein.actions.SHS148k.STRING.txt', type=str,
                    help="ppi path")
parser.add_argument('--pseq_path', default='./data/protein.SHS148k.sequences.dictionary.tsv', type=str,
                    help="protein sequence path")
parser.add_argument('--vec_path', default='./data/vec5_CTC.txt', type=str,
                    help='protein sequence vector path')
parser.add_argument('--split_new', default=False, type=boolean_string,
                    help='split new index file or not')
parser.add_argument('--split_mode', default='random', type=str,
                    help='split method, random, bfs or dfs')
parser.add_argument('--train_valid_index_path', default='./train_valid_index_json/.train_index_random_6', type=str,
                    help='cnn_rnn and gnn unified train and valid ppi index')
parser.add_argument('--use_lr_scheduler', default=True, type=boolean_string,
                    help='train use learning rate scheduler or not')
parser.add_argument('--save_path', default='save_model', type=str,
                    help='model save path')
parser.add_argument('--graph_only_train', default=False, type=boolean_string,
                    help='train ppi graph construct by train or all(train with test)')
parser.add_argument('--batch_size', default=2048, type=int,
                    help='gnn train batch size, edge batch size')
parser.add_argument('--epochs', default=300, type=int,
                    help='train epoch number')
parser.add_argument('--device', default='cuda:0', type=str,
                    help='cuda device, i.e. cuda:0 or cuda:1')
parser.add_argument('--tau_prior', default=1.0, type=float,
                    help='temperature scaling for prior matrix')
parser.add_argument('--gin_in_feature', default=256, type=int,
                    help='Feature dimension for GIN layers.')
parser.add_argument('--lr', default=0.001, type=float,
                    help='Learning rate for the optimizer.')


#曲线参数初始化
train_loss = []
val_loss = []
train_acc = []
val_acc = []
train_recall = []
val_recall = []
train_f1 = []
val_f1 = []


def train(model, graph, ppi_list, loss_fn, optimizer, device,
          result_file_path, summary_writer, save_path, args,
          batch_size=2048, epochs=1000, scheduler=None,
          got=False, t=0.26, esm_tensor=None):

    global_step = 0
    global_best_valid_f1 = 0.0
    global_best_valid_f1_epoch = 0
    loss0 = 0
    loss1 = 0
    loss2 = 0
    loss3 = 0
    loss4 = 0
    loss5 = 0
    loss6 = 0
    loss7 = 0
    global_best_BS_f1 = 0.0
    global_best_ES_f1 = 0.0
    global_best_NS_f1 = 0.0
    global_best_label_f1_list = []


    truth_edge_num = graph.edge_index.shape[1] // 2
    best_val_mse = float('inf')
    best_val_mse_epoch = -1


    mseloss = nn.MSELoss()

    for epoch in range(epochs):

        recall_sum = 0.0
        precision_sum = 0.0
        f1_sum = 0.0
        loss_sum = 0.0

        steps = math.ceil(len(graph.train_mask) / batch_size)

        model.train()

        random.shuffle(graph.train_mask)
        random.shuffle(graph.train_mask_got)

        for step in range(steps):
            if step == steps - 1:
                if got:
                    train_edge_id = graph.train_mask_got[step * batch_size:]
                else:
                    train_edge_id = graph.train_mask[step * batch_size:]
            else:
                if got:
                    train_edge_id = graph.train_mask_got[step * batch_size: step * batch_size + batch_size]
                else:
                    train_edge_id = graph.train_mask[step * batch_size: step * batch_size + batch_size]

            if got:
                output, f = model(graph.x, graph.edge_index_got, train_edge_id, graph.p_matrix, graph.fx)
                label = graph.edge_attr_got[train_edge_id]
            else:
                output, f = model(graph.x, graph.edge_index, train_edge_id, graph.p_matrix, graph.fx)
                label = graph.edge_attr_1[train_edge_id]

            label = label.type(torch.FloatTensor).to(device)

            loss1 = loss_fn(output[:, 0], label[:, 0])
            loss2 = loss_fn(output[:, 1], label[:, 1])
            loss3 = loss_fn(output[:, 2], label[:, 2])
            loss4 = loss_fn(output[:, 3], label[:, 3])
            loss5 = loss_fn(output[:, 4], label[:, 4])
            loss6 = loss_fn(output[:, 5], label[:, 5])
            loss7 = loss_fn(output[:, 6], label[:, 6])
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7

    
            if f is not None:
                p_matrix_target = graph.p_matrix.to(device)
                if args.tau_prior != 1.0:
                    # 为避免 log(0) 或 0^(1/tau) 的问题，加入一个极小值
                    p_matrix_temp = p_matrix_target + 1e-9
                    # 应用公式: P' = Softmax(log(P) / tau) 等价于 P'_i = p_i^(1/tau) / sum(p_j^(1/tau))
                    p_matrix_temp = torch.pow(p_matrix_temp, 1.0 / args.tau_prior)
                    # 重新归一化，确保每行的和为1
                    p_matrix_target = F.normalize(p_matrix_temp, p=1, dim=1)

                loss0 = mseloss(f.to(device), p_matrix_target)
                loss = loss + t * loss0
            # ----------------------------------------------------------------

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m = nn.Sigmoid()
            pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

            metrics = Metrictor_PPI(pre_result.cpu().data, label.cpu().data)
            metrics.show_result()
            
            f1_per_label = []
            for i in range(label.shape[1]):
                label_i = label[:, i].cpu().data.numpy()
                pre_result_i = pre_result[:, i].cpu().data.numpy()
                metrics_label = Metrictor_PPI(pre_result_i, label_i, is_binary=True)
                metrics_label.compute_metrics()
                f1_per_label.append(metrics_label.F1)
            
            overall_f1 = metrics.F1

            recall_sum += metrics.Recall
            precision_sum += metrics.Precision
            f1_sum += overall_f1
            loss_sum += loss.item()

            summary_writer.add_scalar('train/loss', loss.item(), global_step)
            summary_writer.add_scalar('train/precision', metrics.Precision, global_step)
            summary_writer.add_scalar('train/recall', metrics.Recall, global_step)
            summary_writer.add_scalar('train/F1', overall_f1, global_step)

            global_step += 1
            print("epoch: {}, Train: label_loss: {}, precision: {}, recall: {}, f1: {}"
                  .format(epoch, step, loss.item(), metrics.Precision, metrics.Recall, metrics.F1))

        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict()},
                   os.path.join(save_path, 'gnn_model_train.ckpt'))
        
        valid_pre_result_list = []
        valid_label_list = []
        valid_loss_sum = 0.0
        model.eval()
        valid_steps = math.ceil(len(graph.val_mask) / batch_size)

        with torch.no_grad():
            for step in range(valid_steps):
                if step == valid_steps - 1:
                    valid_edge_id = graph.val_mask[step * batch_size:]
                else:
                    valid_edge_id = graph.val_mask[step * batch_size: step * batch_size + batch_size]

                output, f = model(graph.x, graph.edge_index, valid_edge_id, graph.p_matrix, graph.fx)
                label = graph.edge_attr_1[valid_edge_id]
                label = label.type(torch.FloatTensor).to(device)

                loss1 = loss_fn(output[:, 0], label[:, 0])
                loss2 = loss_fn(output[:, 1], label[:, 1])
                loss3 = loss_fn(output[:, 2], label[:, 2])
                loss4 = loss_fn(output[:, 3], label[:, 3])
                loss5 = loss_fn(output[:, 4], label[:, 4])
                loss6 = loss_fn(output[:, 5], label[:, 5])
                loss7 = loss_fn(output[:, 6], label[:, 6])
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7

                if f is not None:
                    p_matrix_target = graph.p_matrix.to(device)
                    if args.tau_prior != 1.0:
                        p_matrix_temp = p_matrix_target + 1e-9
                        p_matrix_temp = torch.pow(p_matrix_temp, 1.0 / args.tau_prior)
                        p_matrix_target = F.normalize(p_matrix_temp, p=1, dim=1)
                    loss0 = mseloss(f.to(device), p_matrix_target)
                    loss = loss + t * loss0

                valid_loss_sum += loss.item()

                summary_writer.add_scalar("Val/MSE", loss0.item(), epoch)
                if loss0.item() < best_val_mse:
                    best_val_mse = loss0.item()
                    best_val_mse_epoch = epoch
                    best_f = f.detach().cpu()
                
                m = nn.Sigmoid()
                pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

                valid_pre_result_list.append(pre_result.cpu().data)
                valid_label_list.append(label.cpu().data)

        BS_list, ES_list, NS_list = BS_ES_NS(ppi_list, graph.train_mask, graph.val_mask)
        valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0)
        metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list)
        metrics.show_result()
        
        f1_per_label = []
        for i in range(valid_label_list.shape[1]):
            label_i = valid_label_list[:, i].cpu().data.numpy()
            pre_result_i = valid_pre_result_list[:, i].cpu().data.numpy()
            metrics_label = Metrictor_PPI(pre_result_i, label_i, is_binary=True)
            metrics_label.compute_metrics()
            f1_per_label.append(metrics_label.F1)
        for i, f1 in enumerate(f1_per_label):
            print(f"Label {i} F1 Score: {f1:.4f}")
        
        overall_f1 = metrics.F1
        
        recall = recall_sum / steps
        precision = precision_sum / steps
        f1 = f1_sum / steps
        loss = loss_sum / steps
        valid_loss = valid_loss_sum / valid_steps

        if scheduler is not None:
            scheduler.step(loss)
            print_file("epoch: {}, now learning rate: {}".format(epoch, scheduler.optimizer.param_groups[0]['lr']),
                       save_file_path=result_file_path)
        
        BS_F1_score, ES_F1_score, NS_F1_score = BS_ES_NS_metrics(valid_pre_result_list, valid_label_list, BS_list, ES_list, NS_list)
        
        if global_best_valid_f1 < overall_f1:
            global_best_valid_f1 = overall_f1
            global_best_valid_f1_epoch = epoch
            global_best_BS_f1 = BS_F1_score
            global_best_ES_f1 = ES_F1_score
            global_best_NS_f1 = NS_F1_score
            global_best_label_f1_list = f1_per_label.copy()
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict()},
                       os.path.join(save_path, 'gnn_model_valid_best.ckpt'))
        
        print("=== Best Epoch Metrics Summary ===")
        print(f"Best Validation F1: {global_best_valid_f1:.4f} (Epoch {global_best_valid_f1_epoch})")
        print(f"Best BS_F1: {global_best_BS_f1:.4f}, ES_F1: {global_best_ES_f1:.4f}, NS_F1: {global_best_NS_f1:.4f}")
        print("Per-label F1 scores at best epoch:")
        for i, f1_score in enumerate(global_best_label_f1_list):
            print(f"  Label {i}: F1 = {f1_score:.4f}")
        
        summary_writer.add_scalar('valid/precision', metrics.Precision, global_step)
        summary_writer.add_scalar('valid/recall', metrics.Recall, global_step)
        summary_writer.add_scalar('valid/F1', overall_f1, global_step)
        summary_writer.add_scalar('valid/loss', valid_loss, global_step)

        print_file(
            "epoch: {}, Training_avg: label_loss: {}, recall: {}, precision: {}, F1: {}, Validation_avg: loss: {}, recall: {},precision: {}, F1: {}, Best_valid_f1: {}, in () epoch"
            .format(epoch, loss, recall, precision, f1, valid_loss, metrics.Recall, metrics.Precision, metrics.F1,
                global_best_valid_f1, global_best_valid_f1_epoch), save_file_path=result_file_path)

        train_loss.append(loss)
        val_loss.append(valid_loss)
        train_acc.append(precision)
        val_acc.append(metrics.Precision)
        train_recall.append(recall)
        val_recall.append(metrics.Recall)
        train_f1.append(f1)
        val_f1.append(metrics.F1)


def main():
    start_time = time.time()
    args = parser.parse_args()

    ppi_data = GNN_DATA(ppi_path=args.ppi_path)
    ppi_data.get_feature_origin(pseq_path=args.pseq_path, vec_path=args.vec_path)
    

    ppi_data.split_dataset(args.train_valid_index_path, random_new=args.split_new, mode=args.split_mode)
    
    ppi_data.generate_data()
    graph = ppi_data.data
    
    #esm_feature_path = "./SHS148k_esm2_embeddings.pt"
    #esm_dict = torch.load(esm_feature_path)

    ppi_list = ppi_data.ppi_list
    graph.train_mask = ppi_data.ppi_split_dict['train_index']
    graph.val_mask = ppi_data.ppi_split_dict['valid_index']
    
    graph.edge_index_got = torch.cat(
        (graph.edge_index[:, graph.train_mask], graph.edge_index[:, graph.train_mask][[1, 0]]), dim=1)
    graph.edge_attr_got = torch.cat((graph.edge_attr_1[graph.train_mask], graph.edge_attr_1[graph.train_mask]), dim=0)
    graph.train_mask_got = [i for i in range(len(graph.train_mask))]

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    esm_tensor = None # This might cause error if ppi_list contains integer indices
    #protein_names_in_order = sorted(ppi_data.protein_name, key=ppi_data.protein_name.get)
    #esm_tensor = torch.stack([esm_dict[pid] for pid in protein_names_in_order]).to(device)


    graph.to(device)

    esm_dim = esm_tensor.shape[1] if esm_tensor is not None else 1280
    model = GIN_Net2(in_len=2000, in_feature=13, gin_in_feature=args.gin_in_feature, num_layers=1, pool_size=3, cnn_hidden=1, esm_feat_dim=esm_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    

    scheduler = None
    if args.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    loss_fn = nn.BCEWithLogitsLoss().to(device)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    save_path = os.path.join(args.save_path, "gnn_{}_{}".format(args.description, time_stamp))
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        
    result_file_path = os.path.join(save_path, "valid_results.txt")
    config_path = os.path.join(save_path, "config.txt")

    with open(config_path, 'w') as f:
        args_dict = vars(args)
        for key in args_dict:
            f.write(f"{key} = {args_dict[key]}\n")
        f.write('\n')
        f.write("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    summary_writer = SummaryWriter(save_path)

    train(model, graph, ppi_list, loss_fn, optimizer, device,
          result_file_path, summary_writer, save_path, args,
          batch_size=args.batch_size, epochs=args.epochs, scheduler=scheduler,
          got=args.graph_only_train, esm_tensor=esm_tensor)

    summary_writer.close()

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
