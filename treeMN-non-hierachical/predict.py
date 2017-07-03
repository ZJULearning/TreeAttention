import six.moves.cPickle as Pickle
import torch as th
from treeMN import TreeMN
from torch.autograd import Variable
from load_data import QADataset
import params
import math
import sys
import csv


bs = 256

testset = QADataset('/home/xuehongyang/TGIF/standard/test.pkl')

label2answer = None
with open('/home/xuehongyang/TGIF/'
          'standard/label2answer.pkl', 'rb') as l2a:
    label2answer = Pickle.load(l2a)

test_sample = 63102
treemn = TreeMN(4096, 300, 1024, 1000)
test_batches = math.ceil(test_sample // bs)
if params.use_cuda:
    treemn.cuda()
print('Start prediction with batch size %d' % bs)
accuracy_test = 0
treemn.load_state_dict(th.load('./snapshot_non_h'))
treemn.eval()
rr = []
for batch in range(test_batches):
    acc = Variable(th.zeros(1), volatile=True)
    if params.use_cuda:
        acc = acc.cuda()

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
        result = r.max(dim=1)[1]
        acc += th.eq(r.max(dim=1)[1].squeeze(), a).float().sum()
        rr.append(list(result.data.cpu().numpy()[0]).copy())
    accuracy_test += acc.data[0]
    sys.stdout.write('\r %d/%d' % (batch, test_batches))
print('test accuracy = %f' % (accuracy_test / test_sample))
print(len(rr))
print(rr[0:10])

with open('ans_TGIF_treemn-noh.tsv', 'wt') as output:
    output = csv.writer(output, delimiter='\t')
    for x in rr:
        output.writerow([label2answer[x[0]]])
