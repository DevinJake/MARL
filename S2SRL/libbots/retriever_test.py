import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from statistics import mean

class PhraseModel(nn.Module):
    def __init__(self, emb_size, dict_size, hid1_size, hid2_size, hid3_size):
        # Call __init__ function of PhraseModel's parent class (nn.Module).
        super(PhraseModel, self).__init__()

        # self.emb = nn.Embedding(num_embeddings=dict_size, embedding_dim=emb_size)
        self.hid_layer1 = nn.Linear(emb_size, hid1_size)
        self.hid_layer2 = nn.Linear(hid1_size, hid2_size)
        self.hid_layer3 = nn.Linear(hid2_size, hid3_size)
        self.output_layer = nn.Linear(hid3_size, dict_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        output = self.hid_layer1(input)
        output = self.hid_layer2(output)
        output = self.hid_layer3(output)
        output = self.softmax(self.output_layer(output))
        logit, index = torch.max(output, dim=1)
        return logit, index

class RetrieverModel(nn.Module):
    def __init__(self, emb_size, dict_size, device):
        # Call __init__ function of PhraseModel's parent class (nn.Module).
        super(RetrieverModel, self).__init__()
        # With :attr:`padding_idx` set, the embedding vector at
        #         :attr:`padding_idx` is initialized to all zeros. However, notice that this
        #         vector can be modified afterwards, e.g., using a customized
        #         initialization method, and thus changing the vector used to pad the
        #         output. The gradient for this vector from :class:`~torch.nn.Embedding`
        #         is always zero.
        self.emb = nn.Embedding(num_embeddings=dict_size+1, embedding_dim=emb_size, padding_idx=dict_size)
        self.device = device

    def forward(self, index, range):
        query = self.pack_input(index)
        documents = self.pack_input(range)
        cosine_output = torch.cosine_similarity(query, documents, dim=1)
        logsoftmax_output = F.log_softmax(cosine_output, dim=0)
        return logsoftmax_output

    def pack_input(self, indices):
        dict_size = self.emb.weight.shape[0]-1
        if not isinstance(indices, tuple):
            index = indices
            if index >= dict_size or index < 0:
                index = dict_size
            input_v = torch.LongTensor([index]).to(self.device)
            input_v = input_v.cuda()
            r = self.emb(input_v)
            return r
        else:
            list = [dict_size if (i >= dict_size or i < 0) else i for i in range(indices[0], indices[1])]
            input_v = torch.LongTensor(list).to(self.device)
            input_v = input_v.cuda()
            r = self.emb(input_v)
            return r

    @classmethod
    def calculate_rank(self, vector):
        rank = 1
        order_list = {}
        if isinstance(vector, list):
            for value in sorted(vector, reverse=True):
                if value not in order_list:
                    order_list[value] = rank
                rank += 1
            order = [order_list[i] for i in vector]
            return order
        else:
            return []

def test1():
    net = PhraseModel(emb_size=50, dict_size=944000, hid1_size=40, hid2_size=30, hid3_size=20).to('cuda')
    net.cuda()
    input = torch.randn(1, 50)
    input = input.cuda()
    print(net(input))

def test2():
    learning_rate = 0.01
    net = RetrieverModel(emb_size=50, dict_size=944000, device='cuda').to('cuda')
    net.cuda()
    net.zero_grad()
    embedding_optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    query_index = [0, 100000, 400000, 600000, 800000]
    document_indices = [(0, 300000), (0, 300000), (300000, 500000), (500000, 700000), (700000, 944000)]
    temp_list = [[13009-0, 34555-0, 234-0, 6789-0, 300000-1-0],
                 [1-0, 6-0, 111111-0, 222222-0, 222223-0],
                 [320000-300000, 330000-300000, 340000-300000, 350000-300000, 360000-300000],
                 [600007-500000, 610007-500000, 620007-500000, 630007-500000, 690007-500000],
                 [700001-700000, 700002-700000, 900000-700000, 910000-700000, 944000-2-700000]]
    for i in range(10000):
        for j in range(len(query_index)):
            embedding_optimizer.zero_grad()
            logsoftmax_output = net(query_index[j], document_indices[j])
            logsoftmax_output = logsoftmax_output.cuda()
            possitive_logsoftmax_output = torch.stack([logsoftmax_output[k] for k in temp_list[j]])
            loss_policy_v = -possitive_logsoftmax_output.mean()
            loss_policy_v = loss_policy_v.cuda()
            loss_policy_v.backward()
            embedding_optimizer.step()
        if i % 100 == 0:
            MAP_list=[]
            for j in range(len(query_index)):
                logsoftmax_output = net(query_index[j], document_indices[j])
                order = net.calculate_rank(logsoftmax_output.tolist())
                orders = [order[k] for k in temp_list[j]]
                if i==0:
                    print('Initial orders for %d:' %j)
                    print(orders)
                MAP = mean(orders)
                MAP_list.append(MAP)
            MAP_for_queries = mean(MAP_list)
            print('%d MAP_for_queries: %f' %(i, MAP_for_queries))
        i += 1

if __name__ == "__main__":
    # test1()
    test2()