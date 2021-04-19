import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Abstract(nn.Module):
    def __init__(self, opt={}):
        super(Abstract, self).__init__()
        self.opt = opt
        self.linear_calculsim = nn.Linear(self.opt['fusion']['dim_h'], 1)

    def _fusion(self, input_v, input_q):
        raise NotImplementedError

    def _calculsim(self, x):
        # x: [batch_size_v, batch_size_t, dim_mm]
        batch_size_v = x.size(0)
        batch_size_t = x.size(1)

        if 'activation' in self.opt['classif']:
            x = getattr(torch, self.opt['classif']['activation'])(x)
        x = F.dropout(x,
                      p=self.opt['classif']['dropout'],
                      training=self.training)
        x = self.linear_calculsim(x)
        x = torch.sigmoid(x)
        x = x.view(batch_size_v, batch_size_t)
        return x

    def forward(self, input_v, input_t):
        mm_sim, v_emb, t_emb, v_dl, t_dl, mm_emb, fake_res = self._fusion(input_v, input_t)
        sim = self._calculsim(mm_sim)
        return sim, v_emb, t_emb, v_dl, t_dl, mm_emb, fake_res


class MMFusion(Abstract):
    def __init__(self, opt={}):
        opt['fusion']['dim_h'] = opt['fusion']['dim_mm']
        super(MMFusion, self).__init__(opt)
        self.fusion = CoreFusion(self.opt)

    def _fusion(self, x_v, x_t):
        return self.fusion(x_v, x_t)


class CoreFusion(nn.Module):
    def __init__(self, opt):
        super(CoreFusion, self).__init__()
        self.opt = opt
        self.class_num = 2
        self.linear_v = nn.Linear(self.opt['fusion']['dim_v'], self.opt['fusion']['dim_hv'])
        self.linear_t = nn.Linear(self.opt['fusion']['dim_t'], self.opt['fusion']['dim_ht'])

        self.bn_layer_v = nn.BatchNorm1d(self.opt['fusion']['dim_hv'], affine=True, track_running_stats=True)
        self.bn_layer_t = nn.BatchNorm1d(self.opt['fusion']['dim_ht'], affine=True, track_running_stats=True)

        self.list_linear_hv = nn.ModuleList([
            nn.Linear(self.opt['fusion']['dim_hv'], self.opt['fusion']['dim_mm'])
            for i in range(self.opt['fusion']['R'])])

        self.list_linear_ht = nn.ModuleList([
            nn.Linear(self.opt['fusion']['dim_ht'], self.opt['fusion']['dim_mm'])
            for i in range(self.opt['fusion']['R'])])

        # self.bn_list_hv = nn.ModuleList(
        #     nn.BatchNorm1d(self.opt['fusion']['dim_hv'], affine=True, track_running_stats=True)
        #     for i in range(self.opt['fusion']['R']))
        # self.bn_list_ht = nn.ModuleList(
        #     nn.BatchNorm1d(self.opt['fusion']['dim_ht'], affine=True, track_running_stats=True)
        #     for i in range(self.opt['fusion']['R']))

        self.dist_learning_v = nn.Linear(self.opt['fusion']['dim_hv'], self.opt['fusion']['dim_hv'])
        # self.dist_learning_t = nn.Linear(self.opt['fusion']['dim_ht'], self.opt['fusion']['dim_ht'])

        self.fake_ln1 = nn.Linear(self.opt['fusion']['dim_mm'], self.opt['fake_dec']['hidden1'])
        self.fake_ln2 = nn.Linear(self.opt['fake_dec']['hidden1'], self.opt['fake_dec']['hidden2'])
        self.fake_ln3 = nn.Linear(self.opt['fake_dec']['hidden2'], self.opt['fake_dec']['hidden3'])
        self.fake_last = nn.Linear(self.opt['fake_dec']['hidden3'], self.class_num)

        self.bn_layer1 = nn.BatchNorm1d(self.opt['fake_dec']['hidden1'], affine=True, track_running_stats=True)
        self.bn_layer2 = nn.BatchNorm1d(self.opt['fake_dec']['hidden2'], affine=True, track_running_stats=True)
        self.bn_layer3 = nn.BatchNorm1d(self.opt['fake_dec']['hidden3'], affine=True, track_running_stats=True)
        self.bn_layer4 = nn.BatchNorm1d(self.class_num, affine=True, track_running_stats=True)

    def forward(self, input_v, input_t):
        batch_size_v = input_v.size(0)
        batch_size_t = input_t.size(0)

        # x_v = F.dropout(input_v, p=self.opt['fusion']['dropout_v'], training=self.training)
        x_v = self.linear_v(input_v)
        x_v = self.bn_layer_v(x_v)
        if 'activation_v' in self.opt['fusion']:
            x_v = getattr(torch, self.opt['fusion']['activation_v'])(x_v)

        # x_t = F.dropout(input_t, p=self.opt['fusion']['dropout_t'], training=self.training)
        x_t = self.linear_t(input_t)
        x_t = self.bn_layer_t(x_t)
        if 'activation_t' in self.opt['fusion']:
            x_t = getattr(torch, self.opt['fusion']['activation_t'])(x_t)

        x_dl_v = self.dist_learning_v(x_v)
        # x_dl_t = self.dist_learning_t(x_t)
        # x_dl_v = x_v
        x_dl_t = x_t

        # calculate the x_mm
        x_mm = []

        # 内积的方式可能将图像的特征赋予的权重太高了，如果使用拼接就可以由神经网络自己选择给图像特征的权重
        # 检索方法中activation_hv都没有，说明这里全都没有做激活，不就相当于一个线性层了啊．．．．．．．
        for i in range(self.opt['fusion']['R']):

            x_hv = F.dropout(x_v, p=self.opt['fusion']['dropout_hv'], training=self.training)
            x_hv = self.list_linear_hv[i](x_hv)
            # x_hv = self.bn_list_hv[i](x_hv)
            if 'activation_hv' in self.opt['fusion']:
                x_hv = getattr(torch, self.opt['fusion']['activation_hv'])(x_hv)

            x_ht = F.dropout(x_t, p=self.opt['fusion']['dropout_ht'], training=self.training)
            x_ht = self.list_linear_ht[i](x_ht)
            # x_ht = self.bn_list_ht[i](x_ht)
            if 'activation_ht' in self.opt['fusion']:
                x_ht = getattr(torch, self.opt['fusion']['activation_ht'])(x_ht)

            # x_hv: [batch_size, dim_mm] ---> [batch_size,  1,          dim_mm]
            # x_ht: [batch_size, dim_mm] ---> [1,           batch_size, dim_mm]
            # torch.mul(): [batch_size(image), batch_size(text), dim_mm]
            x_mm.append(torch.mul(x_hv[:, None, :], x_ht[None, :, :]))  # 'None' create a dim

        x_mm = torch.stack(x_mm, dim=1)  # x_mm: [batch_size(image), R, batch_size(text), dim_mm]
        x_mm = x_mm.sum(1).view(batch_size_v, batch_size_t, self.opt['fusion']['dim_mm'])

        # if 'activation_mm' in self.opt['fusion']:
        #     x_mm = getattr(torch, self.opt['fusion']['activation_mm'])(x_mm)

        # calculate the label of input x_mm
        pairs_num = min(batch_size_v, batch_size_t)
        x_mm_diagonal = x_mm[torch.arange(pairs_num), torch.arange(pairs_num), :]

        # 经过实验证实，在这里分类的阶段，使用bn后再用dropout会带来性能下降．
        # 仅使用bn，或者仅使用dropout就好．
        # fake1 = F.dropout(x_mm_diagonal, p=self.opt['fake_dec']['dropout1'])
        fake1 = self.fake_ln1(x_mm_diagonal)
        fake1 = self.bn_layer1(fake1)
        if 'activation1' in self.opt['fake_dec']:
            fake1 = getattr(torch, self.opt['fake_dec']['activation1'])(fake1)

        # fake2 = F.dropout(fake1, p=self.opt['fake_dec']['dropout2'])
        fake2 = self.fake_ln2(fake1)
        fake2 = self.bn_layer2(fake2)
        if 'activation2' in self.opt['fake_dec']:
            fake2 = getattr(torch, self.opt['fake_dec']['activation2'])(fake2)

        # fake3 = F.dropout(fake2, p=self.opt['fake_dec']['dropout3'])
        fake3 = self.fake_ln3(fake2)
        fake3 = self.bn_layer3(fake3)
        if 'activation3' in self.opt['fake_dec']:
            fake3 = getattr(torch, self.opt['fake_dec']['activation3'])(fake3)

        # fake_res = F.dropout(fake2, p=self.opt['fake_dec']['dropout3'])
        fake_res = self.fake_last(fake3)
        fake_res = self.bn_layer4(fake_res)
        if 'activation4' in self.opt['fake_dec']:  # torch.XXXX()
            fake_res = getattr(torch, self.opt['fake_dec']['activation4'])(fake_res, dim=1)

        return x_mm, x_v, x_t, x_dl_v, x_dl_t, x_mm_diagonal, fake_res


def factory(opt, cuda=True):
    opt = copy.copy(opt)
    model = MMFusion(opt)
    if cuda:
        model = model.cuda()
    return model


