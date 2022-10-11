import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from utils2 import *
import random


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        for i in range(config.NUM_LAYER):
            self.__setattr__("layer_{}".format(i), nn.LSTM(config.MODEL_SIZE, config.MODEL_SIZE))

    def forward(self, inputs, lengths):
        config = self.config
        for i in range(config.NUM_LAYER):
            skip = inputs
            tensor = rnn_utils.pack_padded_sequence(inputs, lengths, enforce_sorted=False)
            tensor, (h, c) = getattr(self, "layer_{}".format(i))(tensor)
            tensor = rnn_utils.pad_packed_sequence(tensor)[0]
            tensor = tensor + skip
        tensor = [x[:i] for x, i in zip(torch.unbind(tensor, axis=1), lengths)]
        return tensor, (h, c)


class Attn(nn.Module):
    def __init__(self, config):
        super(Attn, self).__init__()
        self.config = config
        self.Q = nn.Linear(config.ENC_SIZE, config.ATTN_SIZE)
        self.K = nn.Linear(config.ENC_SIZE, config.ATTN_SIZE)
        self.V = nn.Linear(config.ENC_SIZE, config.ATTN_SIZE)
        self.W = nn.Linear(config.ATTN_SIZE, 1)

    def forward(self, q, k, v, mask):
        config = self.config
        q = self.Q(q)
        k = self.K(k)
        v = self.V(v)
        q = q.unsqueeze(1)
        k = k.unsqueeze(0)
        attn_weight = self.W(torch.tanh(q + k))
        attn_weight = attn_weight.squeeze(-1)
        _nan = torch.tensor(-1e6).to(q.device)
        attn_weight = torch.where(mask, attn_weight, _nan)
        attn_weight = attn_weight.softmax(1)
        attn_weight = attn_weight.unsqueeze(-1)
        context = attn_weight * v
        context = context.sum(1)
        return context


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.attn1 = Attn(config)
        self.attn2 = Attn(config)
        for i in range(config.NUM_LAYER - 1):
            self.__setattr__("layer_{}".format(i), nn.LSTM(config.DEC_SIZE, config.DEC_SIZE))
        self.rnn = nn.LSTM(2 * config.DEC_SIZE, config.DEC_SIZE)

    def forward(self, inputs, l_states, enc1, enc2):
        config = self.config
        lengths1_enc = [x.shape[0] for x in enc1]
        enc1 = rnn_utils.pad_sequence(enc1)
        mask1 = [torch.ones(x).to(enc1.device) for x in lengths1_enc]
        mask1 = rnn_utils.pad_sequence(mask1)
        mask1 = mask1.unsqueeze(0)
        mask1 = mask1.eq(1)
        lengths2_enc = [x.shape[0] for x in enc2]
        enc2 = rnn_utils.pad_sequence(enc2)
        mask2 = [torch.ones(x).to(enc2.device) for x in lengths2_enc]
        mask2 = rnn_utils.pad_sequence(mask2)
        mask2 = mask2.unsqueeze(0)
        mask2 = mask2.eq(1)

        lengths_input = [x.shape[0] for x in inputs]
        tensor = rnn_utils.pad_sequence(inputs)
        for i in range(config.NUM_LAYER - 1):
            skip = tensor
            tensor = rnn_utils.pack_padded_sequence(tensor, lengths_input, enforce_sorted=False)
            tensor, l_states[i] = getattr(self, "layer_{}".format(i))(tensor, l_states[i])
            tensor = rnn_utils.pad_packed_sequence(tensor)[0]
            tensor = tensor + skip

        context1 = self.attn1(tensor, enc1, enc1, mask1)
        context2 = self.attn2(tensor, enc2, enc2, mask2)
        context = context1 + context2
        tensor = torch.cat([tensor, context], -1)
        tensor = rnn_utils.pack_padded_sequence(tensor, lengths_input, enforce_sorted=False)
        tensor, l_states[-1] = self.rnn(tensor, l_states[-1])
        tensor = rnn_utils.pad_packed_sequence(tensor)[0]
        return tensor, l_states

    def beam(self, inputs, l_states, enc1, enc2, size):
        config = self.config
        lengths1_enc = [x.shape[0] for x in enc1]
        enc1 = rnn_utils.pad_sequence(enc1)
        mask1 = [torch.ones(x).to(enc1.device) for x in lengths1_enc]
        mask1 = rnn_utils.pad_sequence(mask1)
        mask1 = mask1.unsqueeze(0)
        mask1 = mask1.eq(1)

        lengths2_enc = [x.shape[0] for x in enc2]
        enc2 = rnn_utils.pad_sequence(enc2)
        mask2 = [torch.ones(x).to(enc2.device) for x in lengths2_enc]
        mask2 = rnn_utils.pad_sequence(mask2)
        mask2 = mask2.unsqueeze(0)
        mask2 = mask2.eq(1)

        enc1 = enc1.unsqueeze(2).repeat(1, 1, size, 1).view(enc1.shape[0], -1, enc1.shape[-1])
        enc2 = enc2.unsqueeze(2).repeat(1, 1, size, 1).view(enc2.shape[0], -1, enc2.shape[-1])
        mask1 = mask1.unsqueeze(-1).repeat(1, 1, 1, size).view(mask1.shape[0], mask1.shape[1], -1)
        mask2 = mask2.unsqueeze(-1).repeat(1, 1, 1, size).view(mask2.shape[0], mask2.shape[1], -1)

        h, c = l_states[0]
        h = h.unsqueeze(2).repeat(1, 1, size, 1).view(h.shape[0], -1, h.shape[-1])
        c = c.unsqueeze(2).repeat(1, 1, size, 1).view(c.shape[0], -1, c.shape[-1])
        l_states = [(h, c) for _ in range(config.NUM_LAYER + 1)]

        lengths_input = [x.shape[0] for x in inputs]
        tensor = rnn_utils.pad_sequence(inputs)
        for i in range(config.NUM_LAYER - 1):
            skip = tensor
            tensor = rnn_utils.pack_padded_sequence(tensor, lengths_input, enforce_sorted=False)
            tensor, l_states[i] = getattr(self, "layer_{}".format(i))(tensor, l_states[i])
            tensor = rnn_utils.pad_packed_sequence(tensor)[0]
            tensor = tensor + skip

        context1 = self.attn1(tensor, enc1, enc1, mask1)
        context2 = self.attn2(tensor, enc2, enc2, mask2)
        context = context1 + context2
        tensor = torch.cat([tensor, context], -1)
        tensor = rnn_utils.pack_padded_sequence(tensor, lengths_input, enforce_sorted=False)
        tensor, l_states[-1] = self.rnn(tensor, l_states[-1])
        tensor = rnn_utils.pad_packed_sequence(tensor)[0]
        return tensor, l_states


class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.embedding_code = nn.Embedding(config.DICT_SIZE_1, config.MODEL_SIZE)
        self.embedding_ast = nn.Embedding(config.DICT_SIZE_2, config.MODEL_SIZE)
        self.embedding_dec = nn.Embedding(config.DICT_SIZE_1, config.MODEL_SIZE)
        self.dropout_enc1 = nn.Dropout(config.DROPOUT)
        self.dropout_enc2 = nn.Dropout(config.DROPOUT)
        self.dropout_h = nn.Dropout(config.DROPOUT)
        self.dropout_c = nn.Dropout(config.DROPOUT)
        self.encoder1 = Encoder(config)
        self.encoder2 = Encoder(config)
        self.decoder = Decoder(config)
        self.linear1 = nn.Linear(2 * config.DEC_SIZE, self.config.DEC_SIZE)
        self.linear2 = nn.Linear(2 * config.DEC_SIZE, self.config.DEC_SIZE)
        self.fc = nn.Linear(config.MODEL_SIZE, config.ACTION_NUM)

    def forward(self, inputs):
        device = self.device
        config = self.config
        in1, in2 = inputs
        lengths1 = [len(x) for x in in1]
        lengths2 = [len(x) for x in in2]
        in1 = [torch.tensor(x).to(device) for x in in1]
        in2 = [torch.tensor(x).to(device) for x in in2]
        in1 = rnn_utils.pad_sequence(in1)
        tensor1 = self.embedding_code(in1)
        in2 = rnn_utils.pad_sequence(in2)
        tensor2 = self.embedding_ast(in2)
        enc1, (h1, c1) = self.encoder1(tensor1, lengths1)
        enc2, (h2, c2) = self.encoder2(tensor2, lengths2)
        enc1 = [self.dropout_enc1(x) for x in enc1]
        enc2 = [self.dropout_enc2(x) for x in enc2]
        l_states_h = torch.cat([h1, h2], -1)
        l_states_h = self.linear1(l_states_h)
        l_states_h = F.relu(l_states_h)
        l_states_c = torch.cat([c1, c2], -1)
        l_states_c = self.linear2(l_states_c)
        l_states_c = F.relu(l_states_c)
        l_states_h = self.dropout_h(l_states_h)
        l_states_c = self.dropout_c(l_states_c)
        l_states = [(l_states_h, l_states_c) for _ in range(config.NUM_LAYER)]

        batch_size = tensor1.shape[1]
        start_token = config.START_TOKEN
        end_token = config.END_TOKEN
        preds = [[start_token] for _ in range(batch_size)]
        tensor = torch.tensor([start_token for _ in range(batch_size)]).to(device)
        outputs = torch.tensor([]).to(device)
        for i in range(config.MAX_OUTPUT_SIZE):
            tensor = tensor.view(1, -1)
            tensor = self.embedding_dec(tensor)
            tensor = tensor.unbind(1)
            tensor, l_states = self.decoder(tensor, l_states, enc1, enc2)
            tensor = self.fc(tensor)
            tensor = tensor.softmax(-1)
            outputs = torch.cat([outputs, tensor], 0)
            for j in range(batch_size):
                if preds[j][-1] != end_token:
                    epsilon = random.random()
                    if epsilon <= 0.8:
                        action_set = tensor.argmax(-1)
                        next_action = action_set[0, j]
                    else:
                        next_action = torch.LongTensor(random.sample(range(tensor.size(2)), 1))
                    if i == 0 and next_action == end_token:
                        tensor = tensor[:, :, :-1]
                        if epsilon <= 0.8:
                            action_set = tensor.argmax(-1)
                            next_action = action_set[0, j]
                        else:
                            next_action = torch.LongTensor(random.sample(range(tensor.size(2) - 1), 1))
                    preds[j].append(int(next_action))
            tensor = torch.tensor([preds[j][-1] for j in range(batch_size)]).to(device)
        preds = [x[1:] for x in preds]
        preds = [x[:-1] if x[-1] == end_token else x for x in preds]
        return preds, outputs

    def pretrain(self, inputs, targets):
        device = self.device
        config = self.config
        in1, in2 = inputs
        lengths1 = [len(x) for x in in1]
        lengths2 = [len(x) for x in in2]
        in1 = [torch.tensor(x).to(device) for x in in1]
        in2 = [torch.tensor(x).to(device) for x in in2]
        in1 = rnn_utils.pad_sequence(in1)
        tensor1 = self.embedding_code(in1)
        in2 = rnn_utils.pad_sequence(in2)
        tensor2 = self.embedding_ast(in2)
        enc1, (h1, c1) = self.encoder1(tensor1, lengths1)
        enc2, (h2, c2) = self.encoder2(tensor2, lengths2)
        enc1 = [self.dropout_enc1(x) for x in enc1]
        enc2 = [self.dropout_enc2(x) for x in enc2]
        l_states_h = torch.cat([h1, h2], -1)
        l_states_h = self.linear1(l_states_h)
        l_states_h = F.relu(l_states_h)
        l_states_c = torch.cat([c1, c2], -1)
        l_states_c = self.linear2(l_states_c)
        l_states_c = F.relu(l_states_c)
        l_states_h = self.dropout_h(l_states_h)
        l_states_c = self.dropout_c(l_states_c)
        l_states = [(l_states_h, l_states_c) for _ in range(config.NUM_LAYER)]

        tensor = [torch.tensor([config.START_TOKEN] + x).to(device) for x in targets]
        tensor = [self.embedding_dec(x) for x in tensor]
        tensor, l_states = self.decoder(tensor, l_states, enc1, enc2)
        tensor = self.fc(tensor)
        return targets, tensor

    def translate(self, inputs):
        device = self.device
        config = self.config
        in1, in2 = inputs
        lengths1 = [len(x) for x in in1]
        lengths2 = [len(x) for x in in2]
        in1 = [torch.tensor(x).to(device) for x in in1]
        in2 = [torch.tensor(x).to(device) for x in in2]
        in1 = rnn_utils.pad_sequence(in1)
        tensor1 = self.embedding_code(in1)
        in2 = rnn_utils.pad_sequence(in2)
        tensor2 = self.embedding_ast(in2)
        enc1, (h1, c1) = self.encoder1(tensor1, lengths1)
        enc2, (h2, c2) = self.encoder2(tensor2, lengths2)
        l_states_h = torch.cat([h1, h2], -1)
        l_states_h = self.linear1(l_states_h)
        l_states_h = F.relu(l_states_h)
        l_states_c = torch.cat([c1, c2], -1)
        l_states_c = self.linear2(l_states_c)
        l_states_c = F.relu(l_states_c)
        l_states = [(l_states_h, l_states_c) for _ in range(config.NUM_LAYER + 1)]
        batch_size = tensor1.shape[1]
        start_token = config.START_TOKEN
        end_token = config.END_TOKEN
        preds = [[start_token] for _ in range(batch_size)]
        tensor = torch.tensor([start_token for _ in range(batch_size)]).to(device)
        for i in range(config.MAX_OUTPUT_SIZE):
            tensor = tensor.view(1, -1)
            tensor = self.embedding_dec(tensor)
            tensor = tensor.unbind(1)
            tensor, l_states = self.decoder(tensor, l_states, enc1, enc2)
            tensor = self.fc(tensor)
            for j in range(batch_size):
                if preds[j][-1] != end_token:
                    action_set = tensor.argmax(-1).detach()
                    next_action = action_set[0, j]
                    if i == 0 and next_action == end_token:
                        tensor = tensor[:, :, :-1]
                        action_set = tensor.argmax(-1)
                        next_action = action_set[0, j]
                    preds[j].append(int(next_action))
            tensor = torch.tensor([preds[j][-1] for j in range(batch_size)]).to(device)
        preds = [x[1:] for x in preds]
        return preds

    def beam_search(self, inputs, size):
        device = self.device
        config = self.config
        in1, in2 = inputs
        lengths1 = [len(x) for x in in1]
        lengths2 = [len(x) for x in in2]
        in1 = [torch.tensor(x).to(device) for x in in1]
        in2 = [torch.tensor(x).to(device) for x in in2]
        in1 = rnn_utils.pad_sequence(in1)
        tensor1 = self.embedding_code(in1)
        in2 = rnn_utils.pad_sequence(in2)
        tensor2 = self.embedding_ast(in2)
        enc1, (h1, c1) = self.encoder1(tensor1, lengths1)
        enc2, (h2, c2) = self.encoder2(tensor2, lengths2)
        l_states_h = torch.cat([h1, h2], -1)
        l_states_h = self.linear1(l_states_h)
        l_states_h = F.relu(l_states_h)
        l_states_c = torch.cat([c1, c2], -1)
        l_states_c = self.linear2(l_states_c)
        l_states_c = F.relu(l_states_c)
        l_states = [(l_states_h, l_states_c) for _ in range(config.NUM_LAYER + 1)]
        batch_size = tensor1.shape[1]
        start_token = config.START_TOKEN
        end_token = config.END_TOKEN
        preds = [[start_token] for _ in range(batch_size)]
        tensor = torch.tensor([start_token for _ in range(batch_size)]).to(device)
        tensor = tensor.view(1, -1)
        tensor = self.embedding_dec(tensor)
        tensor = tensor.unbind(1)
        tensor, l_states = self.decoder(tensor, l_states, enc1, enc2)
        tensor = self.fc(tensor)
        tensor = tensor[:, :, :-1]
        tensor = tensor[-1].softmax(-1).topk(size, -1)
        probs = torch.log(tensor.values).view(-1, 1)
        indices = tensor.indices.view(-1, 1)
        preds = [[config.START_TOKEN, int(indices[i][0])] for i in range(indices.shape[0])]
        completed = [([], 0) for _ in lengths1]
        tensor = torch.tensor([preds[j][-1] for j in range(len(preds))]).to(device)
        for i in range(config.MAX_OUTPUT_SIZE - 1):
            tensor = tensor.view(1, -1)
            tensor = self.embedding_dec(tensor)
            tensor = tensor.unbind(1)
            tensor, l_states = self.decoder.beam(tensor, l_states, enc1, enc2, size)
            tensor = self.fc(tensor)
            tensor = tensor[-1].softmax(-1).topk(size, -1)
            probs = probs + torch.log(tensor.values)
            probs = probs.view(-1, size * size)
            indices = tensor.indices.view(-1, size * size)
            probs = probs.topk(size, -1)
            tensor = probs.indices
            probs = probs.values.view(-1, 1)
            _preds = []
            for j in range(batch_size):
                for k in range(j * size, j * size + size):
                    index = k - j * size
                    index = int(tensor[j][index])
                    l = j * size + index // size
                    pred = [x for x in preds[l]]
                    if pred[-1] != end_token:
                        pred += [int(indices[j][index])]
                        if pred[-1] == end_token:
                            if len(completed[j][0]) == 0 or completed[j][1] < float(probs[k]):
                                completed[j] = (pred, float(probs[k]))
                    _preds.append(pred)
            preds = _preds
            tensor = torch.tensor([preds[x][-1] for x in range(len(preds))]).to(device)
        _preds = [preds[j * size] if len(completed[j][0]) == 0 or probs[j * size] > completed[j][1] else completed[j][0] for j in range(len(lengths1))]
        preds = [x[1:-1] if x[-1] == config.END_TOKEN else x[1:] for x in preds]
        _preds = [x[1:-1] if x[-1] == config.END_TOKEN else x[1:] for x in _preds]
        return preds, _preds


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.actor = Actor(config)
        self.set_trainer()
        self.to(self.device)

    def forward(self, inputs, mode, contracts=None, size=0):
        if mode:
            return self.train_on_batch(inputs, contracts)
        elif mode == False and size == 0:
            return self.translate(inputs)
        elif mode == False and size != 0:
            return self.beam_search(inputs, size)

    def set_trainer(self):
        config = self.config
        self.optimizer_actor = optim.Adam(params=[{"params": self.actor.parameters()}], lr=config.LR, eps=1e-5)
        self.loss_function_pretrain_actor = nn.CrossEntropyLoss(reduction='sum')

    def train_on_batch(self, inputs, contracts):
        self.train()
        device = self.device
        config = self.config
        batch_size = len(inputs[0])
        preds, prob_action = self.actor(inputs)
        lengths = [len(x) for x in preds]
        prob_action = prob_action[:max(lengths)]

        rlist = []
        for i in range(batch_size):
            res = choose_action(contracts[i], preds[i], True)
            rlist.append(res)
        ref_points = torch.tensor(rlist).to(device)

        ref_points = ref_points.view(1, -1)
        mask = torch.zeros_like(prob_action).to(device)
        for i in range(batch_size):
            for j in range(lengths[i]):
                mask[j, i, preds[i][j]] = 1
        mask = mask.eq(1)
        prob_action = prob_action.to(device)
        tensor = torch.where(mask, prob_action, torch.tensor(1e-5).to(device))
        tensor = tensor.sum(-1)
        tensor = torch.log(tensor)
        tensor = tensor * (-ref_points)
        loss_actor = tensor.sum()
        print('loss:', loss_actor)
        loss_actor = loss_actor / sum(lengths)
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm(self.actor.parameters(), max_norm=20, norm_type=2)
        self.optimizer_actor.step()
        return float(loss_actor)

    def translate(self, inputs):
        self.eval()
        config = self.config
        end_token = config.END_TOKEN
        preds = self.actor.translate(inputs)
        preds = [x[:-1] if x[-1] == end_token else x for x in preds]
        return preds

    def beam_search(self, inputs, size):
        self.eval()
        config = self.config
        end_token = config.END_TOKEN
        preds = self.actor.beam_search(inputs, size)
        preds = [x[:-1] if x[-1] == end_token else x for x in preds]
        return preds

    def pretrain(self, inputs, targets, mode):
        self.optimizer_actor.zero_grad()
        device = self.device
        config = self.config
        in1, in2 = inputs
        batch_size = len(in1)
        if mode == "actor":
            self.actor.train()
            lengths = [len(x) + 1 for x in targets]
            _, prob_action = self.actor.pretrain(inputs, targets)
            targets = [torch.tensor(x + [config.END_TOKEN]).to(device) for x in targets]
            loss_actor = 0.
            prob_action = prob_action.to(device)
            for i in range(batch_size):
                loss_actor = loss_actor + self.loss_function_pretrain_actor(prob_action[:lengths[i], i], targets[i])
            loss_actor = loss_actor / sum(lengths)
            loss_actor.backward()
            self.optimizer_actor.step()
            return float(loss_actor)

    def save(self, path):
        checkpoint = {
            'config': self.config,
            'actor': self.actor,
            'optimizer_actor': self.optimizer_actor,
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor = checkpoint['actor']
        self.optimizer_actor = checkpoint['optimizer_actor']
