import logging

import numpy as np
import torch
import tqdm
from torch import nn
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score


class Net(nn.Module):
    def __init__(self, num_questions, hidden_size, num_layers):
        super(Net, self).__init__()
        self.hidden_dim = hidden_size
        self.layer_dim = num_layers
        self.rnn = nn.RNN(num_questions * 2, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, num_questions)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        out, _ = self.rnn(x, h0)
        res = torch.sigmoid(self.fc(out))
        return res


def process_raw_pred(raw_question_matrix, raw_pred, num_questions: int) -> tuple:

    '''
    torch.nonzero(raw_question_matrix): tensor([[ 0, 14],
        [ 1, 14],
        [ 2, 14],
        [ 3, 14],
        [ 4, 14]])
    '''

    # num_questions 最大questions数量
    questions = torch.nonzero(raw_question_matrix)[1:, 1] % num_questions
    '''
    questions: tensor([14, 14, 14, 14])
    '''
    # 总共有length个交互
    length = questions.shape[0]
    pred = raw_pred[: length]
    # 动态调整为一维向量，保证元素个数不变
    pred = pred.gather(1, questions.view(-1, 1)).flatten()

    truth = torch.nonzero(raw_question_matrix)[1:, 1] // num_questions
    skills = torch.nonzero(raw_question_matrix)[0:, 1] % num_questions
    print("skills:", skills)

    return pred, truth, skills


class DKT():
    def __init__(self, num_questions, hidden_size, num_layers):
        super(DKT, self).__init__()
        self.num_questions = num_questions
        self.dkt_model = Net(num_questions, hidden_size, num_layers)


    def train(self, train_data, test_data=None, *, epoch: int, lr=0.002) -> ...:
        count = 0
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.dkt_model.parameters(), lr)
        train_pred_all = []
        train_truth_all = []
        train_skills_all = []
        for e in range(epoch):
            losses = []

            for batch in tqdm.tqdm(train_data, "Epoch %s" % e):

                integrated_pred = self.dkt_model(batch)
                batch_size = batch.shape[0]
                loss = torch.Tensor([0.0])

                for student in range(batch_size):
                    pred, truth, skills = process_raw_pred(batch[student], integrated_pred[student], self.num_questions)
                    count = count + 1
                    print("student:%s" % student, "pred:%s" % pred, "truth:%s" % truth)
                    if pred.shape[0] != 0:
                        loss += loss_function(pred, truth.float())
                    if e == epoch-1:
                        train_skills_all.append(skills)
                        train_pred_all.append(pred)
                        train_truth_all.append(truth)
                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.mean().item())
            print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))
            if test_data is not None:
                auc = self.eval(test_data)
                print("[Epoch %d] auc: %.6f" % (e, auc))


        count = count / 2
        print("total students count is:%s" % count)
        return train_skills_all, train_pred_all, train_truth_all

    def eval(self, test_data) -> float:
        self.dkt_model.eval()
        y_pred = torch.Tensor([])
        y_truth = torch.Tensor([])
        eval_pred_all = []
        eval_truth_all = []
        eval_skills_all = []
        for batch in tqdm.tqdm(test_data, "evaluating"):
            integrated_pred = self.dkt_model(batch)
            batch_size = batch.shape[0]
            for student in range(batch_size):
                pred, truth, skills = process_raw_pred(batch[student], integrated_pred[student], self.num_questions)
                y_pred = torch.cat([y_pred, pred])
                y_truth = torch.cat([y_truth, truth])
                eval_pred_all.append(pred)
                eval_truth_all.append(truth)
                eval_skills_all.append(skills)
        return roc_auc_score(y_truth.detach().numpy(),
                             y_pred.detach().numpy()), eval_pred_all, eval_truth_all, eval_skills_all

    def save(self, filepath):
        torch.save(self.dkt_model.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.dkt_model.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
