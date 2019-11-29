import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder
from models.optimizers import Optimizer
from models.neural import MultiHeadedAttention

def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))


    return optim

def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        print(args.optim)
        print(args.lr_bert)
        print(args.max_grad_norm)
        print(args.beta1)
        print(args.beta2)
        print(args.warmup_steps_bert)
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    # print("optim")
    # print(optim)
    print("named_parameters")
    # for each in model.named_parameters():
    #     print(each)
    # params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    params = [(n, p) for n, p in list(model.named_parameters()) if 'bert.model' in n]
    # print(params)
    optim.set_parameters(params)
    # exit()


    return optim

def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim


def get_generator(vocab_size, dec_hidden_size, device, task):
    if task == 'hybrid':
        gen_func = nn.Softmax(dim=-1)
    else:
        gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator

class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if(large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
        # self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
            self.model = BertModel.from_pretrained('/home/ybai/projects/PreSumm/PreSumm/temp/', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint, lamb=0.8):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.lamb = lamb
        # bert
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        # 抽取层？
        self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)
        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
                                     num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads, intermediate_size=args.ext_ff_size)
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings




        self.W_cont = nn.Parameter(torch.Tensor(1 ,self.bert.model.config.hidden_size))
        self.W_sim = nn.Parameter(torch.Tensor(self.bert.model.config.hidden_size, self.bert.model.config.hidden_size))
        # 重要度的内容可以
        self.W_rel = nn.Parameter(torch.Tensor(self.bert.model.config.hidden_size, self.bert.model.config.hidden_size))
        self.W_doc = nn.Parameter(torch.Tensor(self.bert.model.config.hidden_size, self.bert.model.config.hidden_size))
        self.W_novel = nn.Parameter(torch.Tensor(self.bert.model.config.hidden_size, self.bert.model.config.hidden_size))

        self.b_matrix = nn.Parameter(torch.Tensor(1, 1))

        nn.init.xavier_normal_(self.W_cont)
        nn.init.xavier_normal_(self.W_sim)
        nn.init.xavier_normal_(self.W_rel)
        nn.init.xavier_normal_(self.W_doc)
        nn.init.xavier_normal_(self.W_novel)
        nn.init.xavier_normal_(self.b_matrix)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
        self.to(device)


    def cal_matrix0(self, sent_vec, mask_cls):


        mask_cls = mask_cls.unsqueeze(1).float()
        # print("mask_cls ", mask_cls.size())
        # print(mask_cls)
        mask_my_own = torch.bmm(mask_cls.transpose(1,2), mask_cls)
        sent_num = mask_cls.sum(dim=2).squeeze(1)
        d_rep = sent_vec.mean(dim=1).unsqueeze(1).transpose(1,2)
        score_gather = torch.zeros(1, sent_vec.size(1)).to(self.device)
        for i in range(sent_vec.size(0)): #对于每一个batch
            # print("sent_vec ", sent_vec.size())
            # exit()
            Score_Cont = torch.mm(self.W_cont, sent_vec[i].transpose(0, 1))
            tmp_Sim = torch.mm(sent_vec[i], self.W_sim)
            Score_Sim = torch.mm(tmp_Sim, sent_vec[i].transpose(0, 1)) * mask_my_own[i]
            tmp_rel = torch.mm(sent_vec[i], self.W_rel)
            Score_rel = torch.mm(tmp_rel, d_rep[i]).transpose(0, 1)
            # Score_Sim = torch.mm(tmp_Sim, sent_vec.transpose(1, 2)) * mask_my_own[i]
            # print("score_cont ", Score_Cont.size())
            # print(Score_Cont)
            #
            # print("score_sim ", Score_Sim.size())
            # print(Score_Sim)
            # print("score_rel ", Score_rel.size())
            # print(Score_rel)
            q = Score_rel + Score_Cont + Score_Sim + self.b_matrix
            q = q * mask_my_own[i]
            # print("q before ", q.size())
            # print(q)



            # 计算novelty
            # print(sent_vec[i][0].size())
            # print(self.W_novel.size())
            tmp_nov = torch.mm(sent_vec[i][0].unsqueeze(0), self.W_novel)

            # print(q[0].sum())
            # print("第一项")
            # print(tmp_nov.size())
            # print("第二项")
            # print(((q[0].sum() / sent_num[i]) * sent_vec[i][0]).size())
            accumulation = torch.mm(tmp_nov, nn.functional.tanh(((q[0].sum() / sent_num[i]) * sent_vec[i][0]).unsqueeze(0).transpose(0,1)))
            for j, each_row in enumerate(q):
                if j == 0:
                    continue
                # print("q[",j,"] before:", q[j].size())
                # print(q[j])
                # print("accumulation ", accumulation)
                # exit()
                q[j] = (q[j] + accumulation) * mask_cls[i]
                # print("q[",j,"] after:", q[j].size())
                # print(q[j])
                tmp_nov = torch.mm(sent_vec[i][j].unsqueeze(0), self.W_novel)
                accumulation += torch.mm(tmp_nov, nn.functional.tanh(((q[j].sum() / sent_num[i]) * sent_vec[i][j]).unsqueeze(0).transpose(0,1)))
                # print("accumalation1: ",accumulation)


            q = nn.functional.sigmoid(q) * mask_my_own[i]
            # print("q ", q.size())
            # print(q)
                # exit()


            # print("q", q.size())
            # print(q)
            # exit()

            sum_vec = q.sum(dim=0)

            # print("sum_vec = ", sum_vec.size())
            # print(sum_vec)

            D = torch.diag_embed(sum_vec)

            # print("D = ", D.size())
            # print(D)



            # for j in range(sent_vec.size(1)):
            true_dim = int(sent_num[i])
            # print("true_dim = ", int(true_dim))
            tmp_D = D[:true_dim, :true_dim]
            tmp_q = q[:true_dim, :true_dim]
            D_ = torch.inverse(tmp_D)
            I = torch.eye(true_dim).to(self.device)
            # print(tmp_D)
            # exit()
            y = torch.ones(true_dim, 1).to(self.device) * (1.0 / true_dim)
                             # 1.0 / true_dim).
            # print("I")
            # print(I)
            # print("q")
            # print(q)
            # print("D_")
            # print(D_)
            # print("y")
            # print(y)
            Final_score = torch.mm((1 - self.lamb) * torch.inverse(I - self.lamb * torch.mm(tmp_q, D_)), y).transpose(0,1)
            len_ = D.size(0) - true_dim
            tmp_zeros = torch.zeros(1, len_).to(self.device)
            Final_score = torch.cat((Final_score, tmp_zeros), dim=1)

            # print("Final_score", Final_score.size())
            # print(Final_score)
            # exit()
            if i == 0:
                score_gather += Final_score
            else:
                score_gather = torch.cat((score_gather, Final_score), 0)

        return score_gather

    def cal_matrix(self, sent_vec, mask_cls):

        # [batch_size, 1, hidden_size] * [batch_size, hidden_size, sent_num] = [batch_size, 1, sent_num]
        # Score_Cont = torch.bmm(torch.cat([self.W_cont] * sent_vec.size()[0]), sent_vec.transpose(1,2))
        W_cont0 = self.W_cont.expand(sent_vec.size()[0], self.W_cont.size()[0], self.W_cont.size()[1])
        # W_cont0 = self.W_cont
        # print("W_cont0: ", W_cont0.size())
        # print(W_cont0)
        # print("sent_vec.transpose: ", sent_vec.transpose(1,2).size())
        # print(sent_vec.transpose(1,2))
        mask_cls = mask_cls.unsqueeze(1).float()
        mask_my_own = torch.bmm(mask_cls.transpose(1,2), mask_cls)
        # print("mask_my_ own", mask_my_own.size())
        # print(mask_my_own)

        Score_Cont = torch.bmm(W_cont0, sent_vec.transpose(1,2))
        # exit()
        # print("Score_Cont = ")
        # print(Score_Cont)
        Score_Cont = Score_Cont.expand(Score_Cont.size()[0], sent_vec.size()[1], sent_vec.size()[1]).transpose(1,2)
        # print("Score_Cont = ", Score_Cont.size())
        # print(Score_Cont)

        # [batch_size, sent_num, hidden_size] * [batch_size * hidden_size * hidden_size] = [batch_size, sent_num, hidden_size]
        W_sim0 = self.W_sim.expand(sent_vec.size()[0], self.W_sim.size()[0], self.W_sim.size()[1])
        # tmp_Sim = torch.bmm(sent_vec, torch.cat([self.W_sim] * sent_vec.size()[0]))
        tmp_Sim = torch.bmm(sent_vec, W_sim0)
        # [batch_size, sent_num, hidden_size] * [batch_size, hidden_size, sent_num] = [batch_size, sent_num, sent_num]
        Score_Sim = torch.bmm(tmp_Sim, sent_vec.transpose(1,2)) * mask_my_own

        # print("Score_Sim = ", Score_Sim.size())
        # print(Score_Sim)

        # [batch_size, sent_num, hidden_sum]].mean() = [batch_size, hidden_size, 1]
        d_rep = sent_vec.mean(dim=1).unsqueeze(1).transpose(1,2)


        W_rel0 = self.W_rel.expand(sent_vec.size()[0], self.W_rel.size()[0], self.W_rel.size()[1])
        # [batch_size, sent_num, hidden_size] * [batch_size, hidden_size, hidden_size] = [batch_size, sent_num, hidden_size]
        tmp_rel = torch.bmm(sent_vec, W_rel0)
        # [batch_size, sent_num, hidden_size] * [batch_size, hidden_size, 1] = [batch_size, 1, sent_num]
        Score_rel = torch.bmm(tmp_rel, d_rep).transpose(1,2)
        Score_rel = Score_rel.expand(Score_rel.size()[0], sent_vec.size()[1], sent_vec.size()[1]).transpose(1,2) * mask_my_own
        # print("Score_rel = ", Score_rel.size())
        # print(Score_rel)
        # print("BBB = ", (self.b_matrix.expand(Score_Sim.size()[0],Score_Sim.size()[1], Score_Sim.size()[2]) * mask_my_own).size())
        # print(self.b_matrix.expand(Score_Sim.size()[0],Score_Sim.size()[1], Score_Sim.size()[2]) * mask_my_own)


        q = Score_rel + Score_Cont + Score_Sim + self.b_matrix.expand(Score_Sim.size()[0],Score_Sim.size()[1], Score_Sim.size()[2])
        # print("Final_score = ", Final_score.size())
        # print(Final_score)
        # [batch * sent_num * sent_num]
        q = torch.sigmoid(q) * mask_my_own
        # print("q = ", q.size())
        # print(q)
        # print("Final_score = ", Final_score.size())
        # print(Final_score)
        # batch_size * sent_num * sent_num
        sum_vec = q.sum(dim=1)
        # print("sum_vec = ", sum_vec.size())
        # print(sum_vec)
        D = torch.diag_embed(sum_vec)
        # print("D = ", D.size())
        # print(D)

        # print("sum of mask cls = ", mask_cls.size())
        # print(mask_cls)
        # print(mask_cls.sum(dim=2))
        sent_num = mask_cls.sum(dim=2).squeeze(1)

        # print(sent_num)

        score_gather = torch.zeros(1, Score_Sim.size()[2]).to(self.device)
        for i in range(Score_Sim.size()[0]):
            true_dim = int(sent_num[i])
            # print(int(true_dim))
            tmp_D = D[i][:true_dim, :true_dim]
            tmp_q = q[i][:true_dim, :true_dim]
            # tmp_D = D[i].narrow(0,0,true_dim).narraw(1,0,true_dim)
            # q = q[i].narrow(0, 0, true_dim).narraw(1,0,true_dim)
            D_ = torch.inverse(tmp_D)
            I = torch.eye(true_dim).to(self.device)
            # print(tmp_D)
            # exit()
            y = torch.tensor(1.0 / true_dim).expand(true_dim, 1).to(self.device)
            # print("I")
            # print(I)
            # print("q")
            # print(q)
            # print("D_")
            # print(D_)
            # print("y")
            # print(y)
            Final_score = torch.mm((1 - self.lamb) * torch.inverse(I - self.lamb * torch.mm(tmp_q, D_)), y).transpose(0,1)
            len_ = Score_Sim.size()[2] - true_dim
            tmp_zeros = torch.zeros(1, len_).to(self.device)
            Final_score = torch.cat((Final_score, tmp_zeros), dim=1)
            if i == 0:
                score_gather += Final_score
            else:
                score_gather = torch.cat((score_gather, Final_score), 0)
            # print("Final_score = ", Final_score.size())
            # print(Final_score)
            # exit()



        # D_ = torch.inverse(D)
        # I = torch.eye(Score_Sim.size()[1]).to(self.device)
        # I = I.expand(Score_Sim.size()[0],Score_Sim.size()[1], Score_Sim.size()[2])

        # y = torch.tensor(1.0 / Score_Sim.size()[2]).expand(Score_Sim.size()[0], Score_Sim.size()[2], 1).to(self.device)
        # print("y = ", y.size())
        # print(y)
        #
        # Final_score = torch.bmm((1 - self.lamb) * torch.inverse(I - self.lamb * torch.bmm(q, D_)),  y).transpose(1,2).squeeze(1)
        # print("self.lamb * torch.bmm(q, D) = ", (self.lamb * torch.bmm(q, D_)).size())
        # print(self.lamb * torch.bmm(q, D_))


        # print("score gather = ", score_gather.size())
        # print(score_gather)
        # batch_size * sent_num * sent_num
        return score_gather


    def forward(self, src, segs, clss, mask_src, mask_cls):
        # 先过bert层
        # 第一维为batch_size

        top_vec = self.bert(src, segs, mask_src)
        # print("top vec = ", top_vec.size())
        # print(top_vec)
        # 得到sentence vector
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        # print("sents vec1 = ", sents_vec.size())
        # exit()
        # print(sents_vec)
        sents_vec = sents_vec * mask_cls[:, :, None].float()

        # batchsize * sentencenum * dim
        # print("sents vec2 = ", sents_vec.size())

        # sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)

        # sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)

        sent_scores = self.cal_matrix(sents_vec, mask_cls)
        # print("sent_scores: ",sent_scores.size())
        # print(sent_scores)
        # exit()

        # print(sents_vec)
        # 得到sentence的评分
        # batchsize * sentencenum
        print("sents scores = ", sent_scores.size())
        print(sent_scores)
        exit()
        if self.args.task == "ext":
            return sent_scores, mask_cls
        elif self.args.task == "hybrid":
            return sent_scores, mask_cls, sents_vec


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
                                     intermediate_size=args.enc_ff_size,
                                     hidden_dropout_prob=args.enc_dropout,
                                     attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device, self.args.task)
        self.generator[0].weight = self.decoder.embeddings.weight


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if(args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)

        # exit()
        if self.args.task == "abs":
            decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
            return decoder_outputs, None
        elif self.args.task == 'hybrid':
            decoder_outputs, state, y_embed = self.decoder(tgt[:, :-1], top_vec, dec_state, need_y_emb=True)
            return decoder_outputs, top_vec, y_embed




class HybridSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint = None, checkpoint_ext = None, checkpoint_abs = None):
        super(HybridSummarizer, self).__init__()
        self.args = args
        self.args
        self.device = device


        self.extractor = ExtSummarizer(args, device, checkpoint_ext)
        # self.abstractor = PGTransformers(modules, consts, options)
        self.abstractor = AbsSummarizer(args, device, checkpoint_abs)

        self.context_attn = MultiHeadedAttention(head_count = self.args.dec_heads, model_dim =self.args.dec_hidden_size, dropout=self.args.dec_dropout, need_distribution = True)

        # self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        self.v = nn.Parameter(torch.Tensor(1, self.args.dec_hidden_size * 3))
        self.bv = nn.Parameter(torch.Tensor(1))


        # bert
        # if checkpoint is not None:
        #     self.load_state_dict(checkpoint['model'], strict=True)
        # else:
        #     if args.param_init != 0.0:
        #         for p in self.ext_layer.parameters():
        #             p.data.uniform_(-args.param_init, args.param_init)
        #     if args.param_init_glorot:
        #         for p in self.ext_layer.parameters():
        #             if p.dim() > 1:
        #                 xavier_uniform_(p)

        self.to(device)


    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):


        # print("src = ", src.size())
        # print(src)
        # print("tgt = ", tgt.size())
        # print(tgt)
        # # # # segs是每个词属于哪句话
        # print("segs = ", segs.size())
        # print(segs)
        # # # clss 是每个句子的起点位置
        # print("clss = ", clss.size())
        # print(clss)
        # print("mask_src = ", mask_src.size())
        # print(mask_src)
        # print("mask_cls = ", mask_cls.size())
        # print(mask_cls)

        ext_scores, mask_cls, sent_vec = self.extractor(src, segs, clss, mask_src, mask_cls)
        # exit()

        # batchsize * (tgt_len - 1) * hidden_size
        # 这个隐状态输出出去之后，回到train_abstractive.py的loss计算函数中，然后到loss.py的计算loss函数中，过一个generator的ff层，投影成vocab_size大小的概率分布
        decoder_outputs, encoder_state, y_emb =  self.abstractor(src, tgt, segs, clss, mask_src, mask_tgt, mask_cls)

        # print("encoder_state ", encoder_state.size())
        # print(encoder_state)

        src_len = mask_src.size(-1)
        src_pad_mask = src.eq(0).unsqueeze(1) \
            .expand(src.size(0), tgt.size(-1) - 1, src_len)


        # print("src_pad_mask22222", src_pad_mask.size())
        # print(src_pad_mask)
        context_vector, attn_dist = self.context_attn(encoder_state, encoder_state, decoder_outputs,
                                      mask=src_pad_mask,
                                      # layer_cache=layer_cache,
                                      type="context")

        # print("context_vector ", context_vector.size())
        # # print(context_vector)
        # print("decoder_outputs ", decoder_outputs.size())
        # # print(decoder_outputs)
        # print("y_embed", y_emb.size())
        # print("self.v ", self.v.size())
        # print("self.bv ", self.bv.size())
        # print(tgt[:,:,-1])
        # batch_size * tgt_size * src_size




        # print("11111111111")
        # print(torch.cat([decoder_outputs, y_emb, context_vector], -1).size())
        # print("222222222")
        # print(F.linear(torch.cat([decoder_outputs, y_emb, context_vector], -1),self.v, self.bv))
        ext_dist = torch.zeros(attn_dist.size(0),attn_dist.size(1), self.abstractor.bert.model.config.vocab_size).to(self.device)
        g = torch.sigmoid(F.linear(torch.cat([decoder_outputs, y_emb, context_vector], -1), self.v, self.bv))


        # src: 1 * x_len -> batch * y_len * x_len
        xids = src.unsqueeze(0).repeat(ext_dist.size(1), 1, 1).transpose(0,1)
        # print("ext_dist", ext_dist.size())
        # print("xids", xids.size())
        # print(xids)

        # 留下正常字符，不正常字符的加成为0
        attn_pad_mask = xids.ge(105).float()
            # .expand(tgt_batch, tgt_len, tgt_len)
        # attn_pad_mask_mask = attn_pad_mask

        # torch.set_printoptions(profile="full")
        # print("attn_pad_mask", attn_pad_mask.size())
        # print(attn_pad_mask)
        # torch.set_printoptions(profile="default")

        for i, each_batch in enumerate(clss):
            # print(i)
            for j, each_start in enumerate(each_batch):
                # print(j)
                # 如果已经到边缘,那么直接跳出
                # if clss[i][j + 1] == 0:
                #     break

                # print("extscore", ext_scores[i][j])
                if j + 1 < len(each_batch) and each_batch[j + 1] != 0:
                    # for k in range(each_start + 1, each_batch[j + 1]):
                    #     attn_pad_mask[i][:][k] = ext_scores[i][j]

                    indices = torch.Tensor([k for k in range(each_start + 1, each_batch[j + 1])]).long().to(self.device)

                    # print(indices)
                    # print("scores")
                    # print(ext_scores[i][j])
                    # print("maskkkk")
                    # print(attn_pad_mask[i])

                    attn_pad_mask[i].index_fill_(1, indices, ext_scores[i][j] + 1)
                    # print("attn_pad_maski")
                    # print(attn_pad_mask[i])
                else:
                    indices = torch.Tensor([k for k in range(each_start + 1,  torch.sum(mask_src[i]))]).long().to(self.device)
                    attn_pad_mask[i].index_fill_(1, indices, ext_scores[i][j] + 1)
                    break



                # exit()

        # print("attn_pad_mask", attn_pad_mask.size())
        # print(attn_pad_mask)
        # attn_pad_mask = attn_pad_mask * attn_pad_mask_mask


        # print("ext_scores", ext_scores.size())
        # print(ext_scores)
        # print("ext_ = ", ext_scores.size())
        # print(ext_scores)
        # torch.set_printoptions(profile="full")


        # torch.set_printoptions(profile="default")
        # print("G = ", g.size())
        # print(g)
        # exit()






        # print("attn_dist ", attn_dist.size())
        # print("g = ", g.size())
        # print("xid = ", xids.size())
        # print("ext_dist = ", ext_dist.size())
        # xid 要变成batch * y_len * x_len
        # 然后对应加到ext_dist上边去

        # print("attn_pad_mask", attn_pad_mask.size())
        # print(attn_pad_mask)
        # print("attn_dist ", attn_dist.size())
        # print(attn_dist)
        attn_dist = attn_dist * attn_pad_mask
        ext_vocab_prob = ext_dist.scatter_add(2, xids, (1 - g) * attn_dist)
        # print("ext_vocab_prob")
        # print(ext_vocab_prob.size())
        # exit()
        # print("pred", ext_vocab_prob.size())
        # print(ext_vocab_prob)
        # xids = src[:,:,:-1]




        # print("g = ", g)
        # exit()


        # 算g，现在需要context向量(context vector)、解码器向量(decoder_output)和当前输入向量tgt[::-1]。



        # exit()


        # dec_state = self.abstractor.decoder.init_decoder_state(src, sent_vec)
        # dec_state = self.abstractor.decoder.init_decoder_state(src, sent_vec)
        # decoder_outputs, state = self.abstractor.decoder(tgt[:, :-1], sent_vec, dec_state)
        # exit()










        # print("center_scores = ", center_scores.size())
        # print(center_scores)w
        # exit()
        # y_pred, cost = self.abstractor()
        # src = src[:,:,1:] # 最后一维是从第二个词到clss对应的长度减2的位置
        # clss = clss - 2 * () # 乘一个clss大小的单位矩阵





        '''
        W matrix1: content,不涉及句子之间的关系，可以放在外边的循环里 = c^1_i
        W matrix2: similarity, sent_1^T * W_2 * sent_2 = c^2_ij
        W matrix3: saliece sent_1^T * W_3 * V_doc = c^3_i
        C_ij: \sum_0^j(score_j * sent*j)
        W matrix4: novelty sent_1^T * W_3 * tanh(c_ij) = c^4_ij
        cell_ij = 
        '''
        # exit()
        # 先过bert层
        return decoder_outputs, None, (ext_vocab_prob, g)