import nltk
import six.moves.cPickle as Pickle
import torch as th
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from params import use_cuda


class TreeMN(nn.Module):
    def __init__(self, visual_dim, word_dim,
                 feature_dim, num_output):
        super(TreeMN, self).__init__()
        self.feature_dim = feature_dim
        self.WA = nn.Linear(feature_dim, feature_dim)
        self.WB = nn.Linear(feature_dim, feature_dim)
        self.WQ = nn.Linear(feature_dim, feature_dim, bias=False)
        self.WV = nn.Linear(feature_dim, feature_dim)
        self.WP = nn.Linear(feature_dim, 1)
        self.FC1 = nn.Linear(feature_dim, num_output)
        self.visual_encoder = nn.LSTM(
            input_size=visual_dim,
            hidden_size=feature_dim//2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )
        self.word_dim = word_dim
        self.W = nn.Linear(word_dim, feature_dim)
        self.word_dict = None
        with open('/home/xuehongyang/youtubeclips-dataset/script'
                  '/standard/word_dict.pkl', 'rb') as embed:
            self.word_dict = Pickle.load(embed)

    def init_hidden(self):
        # number_layers*number_directions x batch_size x hidden_size
        return ((Variable(th.zeros(2, 1, self.feature_dim//2)),
                 Variable(th.zeros(2, 1, self.feature_dim//2))))

    # q: feature_dim
    # videos: max_len x feature_dim

    def word_embedding(self, q):
        if q.lower() in self.word_dict.keys():
            r = Variable(
                th.from_numpy(np.asarray(self.word_dict[q.lower()],
                                         dtype=np.float32))
            ).squeeze().unsqueeze(0)
        else:
            r = Variable(
                th.zeros(1, self.word_dim))
        if use_cuda:
            r = r.cuda()
        return self.W(r)

    def attention(self, q, video_embeddings):
        hA = F.tanh(self.WQ(q).expand_as(
            video_embeddings) + self.WV(video_embeddings))
        p = F.softmax(self.WP(hA).squeeze())
        weighted = p.unsqueeze(1).expand_as(
            video_embeddings) * video_embeddings
        v = weighted.sum(dim=0)
        return v + q

    def computeTree(self, tree, videos):
        h = Variable(th.zeros(1, self.feature_dim))
        if use_cuda:
            h = h.cuda()
        for subtree in tree:
            if type(subtree) == nltk.tree.Tree and\
               subtree.height() > 2:
                x = self.computeTree(subtree, videos)
                # if subtree.label()[0] == 'att':
                #     x = self.attention(x, videos)
                #     x = self.WA(x)
                # else:
                #     x = self.WB(x)
                x = self.WB(x)
                h += x
            else:
                # the leaves should be dealt differently
                l = subtree

                for leaf in l:
                    x = self.word_embedding(leaf)
                if l.label()[0] == 'att':
                    x = self.attention(x, videos)
                    x = self.WA(x)
                else:
                    x = self.WB(x)
                h += x

        return F.tanh(h)

    def forward(self, v, sent, v_hidden):
        v = v.unsqueeze(0)
        v_enc, v_hidden = self.visual_encoder(v, v_hidden)
        video = v_enc.squeeze()
        fea = self.computeTree(sent, video)
        return self.FC1(fea)
