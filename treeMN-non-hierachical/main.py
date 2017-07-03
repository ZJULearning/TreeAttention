import torch as th
from treeMN import TreeMN
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from load_data import QADataset
import params
from progressbar import Percentage, Bar, ProgressBar, Timer
import itchat
import datetime
import random
import math


bs = 64
dataset = QADataset('/home/xuehongyang/TGIF/standard/train.pkl')
valset = QADataset('/home/xuehongyang/TGIF/standard/val.pkl')
testset = QADataset('/home/xuehongyang/TGIF/standard/test.pkl')


train_sample = 79971
val_sample = 19526
test_sample = 63102
train_num = list(range(train_sample))
treemn = TreeMN(4096, 300, 1024, 1000)
val_batches = math.ceil(val_sample // bs)
train_batches = math.ceil(train_sample // bs)
test_batches = math.ceil(test_sample // bs)
if params.use_cuda:
    treemn.cuda()
n_epoch = 300
optimizer = th.optim.Adam(treemn.parameters(), lr=0.0001, weight_decay=0.0005)
nn.utils.clip_grad_norm(treemn.parameters(), 10.0)

output_interval = 40

best_test = 0.0
best_epoch = None
# itchat.auto_login(enableCmdQR=2, hotReload=True)
# itchat.send('Training TreeMN model with'
#             ' batch size %d' % (bs),
#             toUserName='filehelper')
print('Start training with batch size %d' % bs)
for epoch in range(n_epoch):
    treemn.train()
    widgets = ['Epoch: %d ' % epoch, Percentage(), ' ', Bar(marker='#'),
               ' ', Timer(), ' ']
    pbar = ProgressBar(widgets=widgets, maxval=dataset.__len__()//bs).start()
    random.shuffle(train_num)
    train_loss = 0
    for batch in range(train_batches):
        loss = Variable(th.zeros(1), requires_grad=True)
        if params.use_cuda:
            loss = loss.cuda()

        optimizer.zero_grad()
        current_bs = 0
        for i in range(bs):
            j = batch * bs + i
            if j >= train_sample:
                break
            current_bs += 1
            v, q, a = dataset.__getitem__(train_num[j])

            v = Variable(v)
            a = Variable(a)

            v_hidden = treemn.init_hidden()
            if params.use_cuda:
                v = v.cuda()
                v_hidden = [x.cuda() for x in v_hidden]
                a = a.cuda()

            r = treemn.forward(v, q, v_hidden)
            loss += F.cross_entropy(r, a).squeeze()

        train_loss += loss.data[0]
        loss = loss / current_bs
        loss.backward()
        optimizer.step()
        pbar.update(batch)
        if batch % output_interval == 0:
            print(' batch %d: loss = %f' %
                  (batch,
                   train_loss / (batch * bs + current_bs)))

    t_loss = train_loss / train_sample
    pbar.finish()
    treemn.eval()
    d1 = datetime.datetime.now()
    accuracy = 0
    for batch in range(val_batches):
        acc = Variable(th.zeros(1), volatile=True)
        if params.use_cuda:
            acc = acc.cuda()

        optimizer.zero_grad()
        for i in range(bs):
            j = batch * bs + i
            if j >= val_sample:
                break
            v, q, a = valset.__getitem__(j)

            v = Variable(v, volatile=True)
            a = Variable(a, volatile=True)

            v_hidden = treemn.init_hidden()
            if params.use_cuda:
                v = v.cuda()
                v_hidden = [x.cuda() for x in v_hidden]
                a = a.cuda()

            r = treemn.forward(v, q, v_hidden)
            acc += th.eq(r.max(dim=1)[1].squeeze(), a).float().sum()
        accuracy += acc.data[0]
    d2 = datetime.datetime.now()
    duration = (d2 - d1).seconds
    print('non-hierachical validation takes %d seconds' % duration)
    print('validation accuracy = %f' % (accuracy / val_sample))

    accuracy_test = 0
    d1 = datetime.datetime.now()
    for batch in range(test_batches):
        acc = Variable(th.zeros(1), volatile=True)
        if params.use_cuda:
            acc = acc.cuda()

        optimizer.zero_grad()
        for i in range(bs):
            j = batch * bs + i
            if j >= test_sample:
                break
            v, q, a = testset.__getitem__(j)

            v = Variable(v, volatile=True)
            a = Variable(a, volatile=True)

            v_hidden = treemn.init_hidden()
            if params.use_cuda:
                v = v.cuda()
                v_hidden = [x.cuda() for x in v_hidden]
                a = a.cuda()

            r = treemn.forward(v, q, v_hidden)
            acc += th.eq(r.max(dim=1)[1].squeeze(), a).float().sum()
        accuracy_test += acc.data[0]
    d2 = datetime.datetime.now()
    duration = (d1 - d2).seconds
    print('test takes %d seconds' % (duration))
    print('test accuracy = %f' % (accuracy_test / test_sample))

    if accuracy_test / test_sample > best_test:
        th.save(treemn.state_dict(),
                './snapshot_non_h')
        best_test = accuracy_test / test_sample
        best_epoch = epoch
    print('current best epoch: %d, test accuracy = %f' % (best_epoch,
                                                          best_test))
    # try:
    #     itchat.send('The TreeMN-1024-non-h model: epoch = %d,'
    #                 ' train loss = %f, validation accuracy = %f,'
    #                 'test accuracy = %f' %
    #                 (epoch, t_loss,
    #                  accuracy / val_sample,
    #                  accuracy_test / test_sample),
    #                 toUserName='filehelper')
    # except:
    #     pass
