import os
import json
import numpy as np
import copy
import torch
import random

from tqdm import tqdm

from utils import UnionFindSet, get_bfs_sub_graph, get_dfs_sub_graph
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader

NP_ZEROS = np.zeros((7, 7))


class GNN_DATA:
    def __init__(self, ppi_path, exclude_protein_path=None, max_len=2000, skip_head=True, p1_index=0, p2_index=1,
                 label_index=2, graph_undirection=True, bigger_ppi_path=None):
        self.ppi_list = []
        self.ppi_dict = {}
        self.ppi_label_list = []
        self.protein_dict = {}
        self.protein_name = {}
        self.ppi_path = ppi_path
        self.bigger_ppi_path = bigger_ppi_path
        self.max_len = max_len

        name = 0
        ppi_name = 0
        # maxlen = 0
        self.node_num = 0
        self.edge_num = 0
        if exclude_protein_path != None:
            with open(exclude_protein_path, 'r') as f:
                ex_protein = json.load(f)
                f.close()
            ex_protein = {p: i for i, p in enumerate(ex_protein)}
        else:
            ex_protein = {}

        class_map = {'reaction': 0, 'binding': 1, 'ptmod': 2, 'activation': 3, 'inhibition': 4, 'catalysis': 5,
                     'expression': 6}

        for line in tqdm(open(ppi_path)):
            if skip_head:
                skip_head = False
                continue
            line = line.strip().split('\t')

            if line[p1_index] in ex_protein.keys() or line[p2_index] in ex_protein.keys():
                continue

            # get node and node name
            if line[p1_index] not in self.protein_name.keys():
                self.protein_name[line[p1_index]] = name
                name += 1

            if line[p2_index] not in self.protein_name.keys():
                self.protein_name[line[p2_index]] = name
                name += 1

            # get edge and its label
            temp_data = ""
            if line[p1_index] < line[p2_index]:
                temp_data = line[p1_index] + "__" + line[p2_index]
            else:
                temp_data = line[p2_index] + "__" + line[p1_index]

            if temp_data not in self.ppi_dict.keys():
                self.ppi_dict[temp_data] = ppi_name
                temp_label = [0, 0, 0, 0, 0, 0, 0]
                temp_label[class_map[line[label_index]]] = 1
                self.ppi_label_list.append(temp_label)
                ppi_name += 1
            else:
                index = self.ppi_dict[temp_data]
                temp_label = self.ppi_label_list[index]
                temp_label[class_map[line[label_index]]] = 1
                self.ppi_label_list[index] = temp_label

        if bigger_ppi_path != None:
            skip_head = True
            for line in tqdm(open(bigger_ppi_path)):
                if skip_head:
                    skip_head = False
                    continue
                line = line.strip().split('\t')

                if line[p1_index] not in self.protein_name.keys():
                    self.protein_name[line[p1_index]] = name
                    name += 1

                if line[p2_index] not in self.protein_name.keys():
                    self.protein_name[line[p2_index]] = name
                    name += 1

                temp_data = ""
                if line[p1_index] < line[p2_index]:
                    temp_data = line[p1_index] + "__" + line[p2_index]
                else:
                    temp_data = line[p2_index] + "__" + line[p1_index]

                if temp_data not in self.ppi_dict.keys():
                    self.ppi_dict[temp_data] = ppi_name
                    temp_label = [0, 0, 0, 0, 0, 0, 0]
                    temp_label[class_map[line[label_index]]] = 1
                    self.ppi_label_list.append(temp_label)
                    ppi_name += 1
                else:
                    index = self.ppi_dict[temp_data]
                    temp_label = self.ppi_label_list[index]
                    temp_label[class_map[line[label_index]]] = 1
                    self.ppi_label_list[index] = temp_label

        i = 0
        for ppi in tqdm(self.ppi_dict.keys()):
            name = self.ppi_dict[ppi]
            assert name == i
            i += 1
            temp = ppi.strip().split('__')
            self.ppi_list.append(temp)

        ppi_num = len(self.ppi_list)
        self.origin_ppi_list = copy.deepcopy(self.ppi_list)
        assert len(self.ppi_list) == len(self.ppi_label_list)
        for i in tqdm(range(ppi_num)):
            seq1_name = self.ppi_list[i][0]
            seq2_name = self.ppi_list[i][1]
            # print(len(self.protein_name))
            self.ppi_list[i][0] = self.protein_name[seq1_name]
            self.ppi_list[i][1] = self.protein_name[seq2_name]

        if graph_undirection:
            for i in tqdm(range(ppi_num)):
                temp_ppi = self.ppi_list[i][::-1]
                temp_ppi_label = self.ppi_label_list[i]
                # if temp_ppi not in self.ppi_list:
                self.ppi_list.append(temp_ppi)
                self.ppi_label_list.append(temp_ppi_label)

        self.node_num = len(self.protein_name)
        self.edge_num = len(self.ppi_list)

    def get_protein_aac(self, pseq_path):
        # aac: amino acid sequences

        self.pseq_path = pseq_path
        self.pseq_dict = {}
        self.protein_len = []

        for line in tqdm(open(self.pseq_path)):
            line = line.strip().split('\t')
            if line[0] not in self.pseq_dict.keys():
                self.pseq_dict[line[0]] = line[1]
                self.protein_len.append(len(line[1]))

        print("protein num: {}".format(len(self.pseq_dict)))
        print("protein average length: {}".format(np.average(self.protein_len)))
        print("protein max & min length: {}, {}".format(np.max(self.protein_len), np.min(self.protein_len)))

    def embed_normal(self, seq, dim):
        if len(seq) > self.max_len:
            return seq[:self.max_len]
        elif len(seq) < self.max_len:
            less_len = self.max_len - len(seq)
            return np.concatenate((seq, np.zeros((less_len, dim))))
        return seq

    def vectorize(self, vec_path):
        self.acid2vec = {}
        self.dim = None
        for line in open(vec_path):
            line = line.strip().split('\t')
            temp = np.array([float(x) for x in line[1].split()])
            self.acid2vec[line[0]] = temp
            if self.dim is None:
                self.dim = len(temp)
        print("acid vector dimension: {}".format(self.dim))

        self.pvec_dict = {}

        for p_name in tqdm(self.pseq_dict.keys()):
            temp_seq = self.pseq_dict[p_name]
            temp_vec = []
            for acid in temp_seq:
                temp_vec.append(self.acid2vec[acid])
            temp_vec = np.array(temp_vec)

            temp_vec = self.embed_normal(temp_vec, self.dim)

            self.pvec_dict[p_name] = temp_vec

    def get_feature_origin(self, pseq_path, vec_path):
        self.get_protein_aac(pseq_path)

        self.vectorize(vec_path)

        self.protein_dict = {}
        for name in tqdm(self.protein_name.keys()):
            self.protein_dict[name] = self.pvec_dict[name]

    def get_connected_num(self):
        self.ufs = UnionFindSet(self.node_num)
        ppi_ndary = np.array(self.ppi_list)
        for edge in ppi_ndary:
            start, end = edge[0], edge[1]
            self.ufs.union(start, end)

    def generate_data(self):
        self.get_connected_num()
    
        print("Connected domain num: {}".format(self.ufs.count))
    
        ppi_list = np.array(self.ppi_list)
        ppi_label_list = np.array(self.ppi_label_list)
    
        self.edge_index = torch.tensor(ppi_list, dtype=torch.long)
        self.edge_attr = torch.tensor(ppi_label_list, dtype=torch.long)
    
        self.x = []
        i = 0
        for name in self.protein_name:
            assert self.protein_name[name] == i
            i += 1
            self.x.append(self.protein_dict[name])
    
        self.x = np.array(self.x)
        self.x = torch.tensor(self.x, dtype=torch.float)
    
        # 确保 ppi_split_dict 已经被 split_dataset 方法填充
        if not hasattr(self, 'ppi_split_dict'):
            raise Exception("Please call split_dataset() before generate_data().")
    
        self.tr_ind = []
        self.ts_ind = []
        self.tr_graph = []
        self.ts_graph = []
        self.tr_label = []
        self.fx = []
        ##得到训练数据的下标
        tr_ind = self.ppi_split_dict['train_index']
        ts_ind = self.ppi_split_dict['valid_index']

        tr_graph = self.edge_index[tr_ind]
        ts_graph = self.edge_index[ts_ind]

        tr_label = self.edge_attr[tr_ind]
        ts_label = self.edge_attr[ts_ind]

        # print('tr_label---------------------------', tr_label)
        fx = [[], [], [], [], [], [], []]
        for i in range(tr_label.shape[0]):
            for j in range(7):
                if tr_label[i, j] == 1:
                    fx[j].extend(tr_graph[i].numpy())
        for i in range(7):
            fx[i] = list(set(fx[i]))
        # print('fx:', fx)
        self.fx = fx #赋值
        # 统计训练集中每种标签的数量和百分比
        label_counts = torch.sum(tr_label, dim=0)  # 每列求和 -> 每种标签的数量
        total_labels = torch.sum(label_counts).item()  # 总的标签数（多个标签可叠加）

        # print("训练集中每种标签的数量和百分比:")
        # for i in range(7):
        #     count = label_counts[i].item()
        #     percent = (count / total_labels) * 100 if total_labels > 0 else 0
        #     print(f"标签 {i}: 数量 = {count}, 百分比 = {percent:.2f}%")
        # print(f"训练集边数: {tr_label.shape[0]}")
        # 输出边数统计
        total_edges = self.edge_index.shape[0]  # 总的边数（包括对称扩展后的）
        train_edges = tr_label.shape[0]
        valid_edges = ts_label.shape[0]

        # print(f"\n[边数统计]")
        # print(f"总边数（含对称边）: {total_edges}")
        # print(f"训练集边数: {train_edges}")
        # print(f"验证集边数: {valid_edges}")

        # 条件概率
        labels = np.array(ppi_label_list)

        label_graph = NP_ZEROS
        labels_tr = labels[self.ppi_split_dict['train_index']]
        for i in range(7):
            sums_query = np.sum(labels_tr[:, i] == 1)
            # print(sums_query)
            sums_where = np.argwhere(labels_tr[:, i] == 1).flatten()
            for j in range(7):
                A = labels_tr[sums_where]
                # print(A)
                xxx = np.sum(A[:, j] == 1)
                # print(xxx)
                label_graph[i, j] = xxx / sums_query

        self.p_matrix = torch.Tensor(label_graph)
        # norm = torch.norm(self.p_matrix, dim=1, keepdim=True)
        # self.p_matrix = self.p_matrix / norm  # 7 * 7
        # print('self.p_matrix:', self.p_matrix)
        # print('self.edge_index:', self.edge_index)
        # print('self.edge_attr:', self.edge_attr)
        #
        sum_counts = [0] * 7
        for i in range(7):
            sum_counts[i] = (self.edge_attr[:, i]).sum().item()
        for i in range(7):
            print(sum_counts[i])


        self.data = Data(x=self.x, edge_index=self.edge_index.T, edge_attr_1=self.edge_attr, p_matrix=self.p_matrix,
                         fx=self.fx)

    def split_dataset(self, train_valid_index_path, test_size=0.2, random_new=False, mode='random'):
        if random_new:
            if mode == 'random':
                ppi_num = int(self.edge_num // 2)
                random_list = [i for i in range(ppi_num)]
                random.shuffle(random_list)

                self.ppi_split_dict = {}
                self.ppi_split_dict['train_index'] = random_list[: int(ppi_num * (1 - test_size))]
                self.ppi_split_dict['valid_index'] = random_list[int(ppi_num * test_size):]

                jsobj = json.dumps(self.ppi_split_dict)
                with open(train_valid_index_path, 'w') as f:
                    f.write(jsobj)
                    f.close()

            elif mode == 'bfs' or mode == 'dfs':
                print("use {} method split train and valid dataset".format(mode))
                node_to_edge_index = {}
                edge_num = int(self.edge_num // 2)
                for i in range(edge_num):
                    edge = self.ppi_list[i]
                    if edge[0] not in node_to_edge_index.keys():
                        node_to_edge_index[edge[0]] = []
                    node_to_edge_index[edge[0]].append(i)

                    if edge[1] not in node_to_edge_index.keys():
                        node_to_edge_index[edge[1]] = []
                    node_to_edge_index[edge[1]].append(i)

                node_num = len(node_to_edge_index)

                sub_graph_size = int(edge_num * test_size)
                if mode == 'bfs':
                    selected_edge_index = get_bfs_sub_graph(self.ppi_list, node_num, node_to_edge_index, sub_graph_size)
                elif mode == 'dfs':
                    selected_edge_index = get_dfs_sub_graph(self.ppi_list, node_num, node_to_edge_index, sub_graph_size)

                all_edge_index = [i for i in range(edge_num)]

                unselected_edge_index = list(set(all_edge_index).difference(set(selected_edge_index)))

                self.ppi_split_dict = {}
                self.ppi_split_dict['train_index'] = unselected_edge_index
                self.ppi_split_dict['valid_index'] = selected_edge_index
                # print('self.train_idnex:', self.train_index)

                assert len(unselected_edge_index) + len(selected_edge_index) == edge_num

                jsobj = json.dumps(self.ppi_split_dict)
                with open(train_valid_index_path, 'w') as f:
                    f.write(jsobj)
                    f.close()

            else:
                print("your mode is {}, you should use bfs, dfs or random".format(mode))
                return
        else:
            with open(train_valid_index_path, 'r') as f:
                self.ppi_split_dict = json.load(f)
                f.close()


if __name__ == '__main__':
    torch.set_printoptions(threshold=float('inf'))  # 打印tensor的时候不打印省略号，将数据完整输出
    ppi_data = GNN_DATA(ppi_path='./data/protein.actions.SHS148k.STRING.txt')
    # ppi_data = GNN_DATA(ppi_path='/apdcephfs/share_1364275/kaithgao/ppi/protein.actions.SHS148k.STRING.txt')
    ppi_data.get_feature_origin(pseq_path='./data/protein.SHS148k.sequences.dictionary.tsv',
                                vec_path='./data/vec5_CTC.txt')
    ppi_data.split_dataset('./train_valid_index_json/.train_index_random_6') # 示例：需要先调用split
    ppi_data.generate_data()
    graph = ppi_data.data
    print(graph)
