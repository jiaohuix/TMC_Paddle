import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from itertools import chain

# loss function
def KL(alpha, c):
    beta = paddle.ones((1, c))
    S_alpha = paddle.sum(alpha, axis=1, keepdim=True)
    S_beta = paddle.sum(beta, axis=1, keepdim=True)
    lnB = paddle.lgamma(S_alpha) - paddle.sum(paddle.lgamma(alpha), axis=1, keepdim=True)
    lnB_uni = paddle.sum(paddle.lgamma(beta), axis=1, keepdim=True) - paddle.lgamma(S_beta)
    dg0 = paddle.digamma(S_alpha)
    dg1 = paddle.digamma(alpha)
    kl = paddle.sum((alpha - beta) * (dg1 - dg0), axis=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = paddle.sum(alpha, axis=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = paddle.sum(label * (paddle.digamma(S) - paddle.digamma(alpha)), axis=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return (A + B)

def mse_loss(p, alpha, c, global_step, annealing_step=1):
    S = paddle.sum(alpha, axis=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = paddle.sum((label - m) ** 2, axis=1, keepdim=True)
    B = paddle.sum(alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    return (A + B) + C


class TMC(nn.Layer):

    def __init__(self, classes, views, classifier_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of views
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(TMC, self).__init__()
        self.views = views
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        self.classifier_dims=list(chain(*classifier_dims))
        self.Classifiers = nn.LayerList([Classifier(classifier_dims[i], self.classes) for i in range(self.views)])

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = paddle.sum(alpha[v], axis=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = self.classes/S[v]

            # b^0 @ b^(0+1)
            bb = paddle.bmm(b[0].reshape([-1, self.classes, 1]), b[1].reshape([-1, 1, self.classes]))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = paddle.multiply(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = paddle.multiply(b[1], uv_expand)
            # calculate C
            bb_sum = paddle.sum(bb, axis=(1, 2)) # [200,10,10]
            bb_diag = paddle.diagonal(bb, axis1=-2, axis2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (paddle.multiply(b[0], b[1]) + bu + ub)/((1-C).reshape([-1, 1]).expand(b[0].shape))
            # calculate u^a
            u_a = paddle.multiply(u[0], u[1])/((1-C).reshape([-1, 1]).expand(u[0].shape))

            # calculate new S
            S_a = self.classes / u_a
            # calculate new e_k
            e_a = paddle.multiply(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha)-1):
            if v==0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a

    def forward(self,x):
        """
        :param input: Multi-view data [bsz,feature_dim_sum]
        :return: evidence of every view  [bsz,views,classes]
        """
        evidence=[]
        for v_num in range(self.views):
            cur_dim=self.classifier_dims[v_num]
            past_dims=sum(self.classifier_dims[:v_num])
            cur_x=x[:,past_dims:past_dims+cur_dim]
            evidence_ = self.Classifiers[v_num](cur_x)
            evidence.append(evidence_)

        alpha = dict()
        for v_num in range(len(evidence)):
            alpha[v_num] = evidence[v_num] + 1
        # step four
        alpha_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1
        return evidence,evidence_a,alpha,alpha_a

    def criterion(self,alpha,alpha_a,y,global_step):
        # step one
        loss = 0
        for v_num in range(len(alpha)):
            # step two
            # step three
            loss += ce_loss(y, alpha[v_num], self.classes, global_step, self.lambda_epochs)
        # step four
        evidence_a = alpha_a - 1
        loss += ce_loss(y, alpha_a, self.classes, global_step, self.lambda_epochs)
        loss = paddle.mean(loss)
        return  loss

    def forward1(self, X, y, global_step):
        '''

        :param X:  input: Multi-view data

        :param y:
        :param global_step:
        :return:
        '''
        # step one
        evidence = self.infer(X)
        loss = 0
        alpha = dict()
        for v_num in range(len(X)):
            # step two
            alpha[v_num] = evidence[v_num] + 1
            # step three
            loss += ce_loss(y, alpha[v_num], self.classes, global_step, self.lambda_epochs)
        # step four
        alpha_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1
        loss += ce_loss(y, alpha_a, self.classes, global_step, self.lambda_epochs)
        loss = paddle.mean(loss)
        return evidence, evidence_a, loss

    def infer(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        evidence = dict()
        for v_num in range(self.views):
            evidence[v_num] = self.Classifiers[v_num](input[v_num])
        return evidence


class Classifier(nn.Layer):
    def __init__(self, classifier_dims, classes):
        super(Classifier, self).__init__()
        self.num_layers = len(classifier_dims)
        self.fc = nn.LayerList()
        for i in range(self.num_layers-1):
            self.fc.append(nn.Linear(classifier_dims[i], classifier_dims[i+1]))
        self.fc.append(nn.Linear(classifier_dims[self.num_layers-1], classes))
        self.fc.append(nn.Softplus())

    def forward(self, x):
        h = self.fc[0](x)
        for i in range(1, len(self.fc)):
            h = self.fc[i](h)
        return h
