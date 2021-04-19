import torch
import numpy as np


def params_count(model):
    count = 0
    for p in model.parameters():
        c = 1
        for i in range(p.dim()):
            c *= p.size(i)
        count += c
    return count


def euclidean_dist(x, y):
    dist = x-y
    dist = torch.pow(dist, 2).sum(1, keepdim=True)
    dist = dist.sqrt()
    return dist.cuda()


def l2norm_r(X):
    """L2-normalize rows of X
    """
    norm = torch.pow(X, 2).sum(dim=0, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def inner_conduct(x):
    temp = torch.zeros(x.shape[0])
    for i in range(x.shape[0]):
        temp[i] = torch.dot(x[i],x[i])
    return temp.cuda()


def cos_sim(x, y, mask):
    # 我不知道下面这个公式是不是正确计算了我想要的东西．．
    sim = torch.nn.functional.cosine_similarity(x[:, None, :], y[None, :, :], dim=2)
    sim = sim.masked_fill_(mask, 0)
    im_neg = sim.max(1)[1]
    te_neg = sim.max(0)[1]
    return im_neg, te_neg


def e_sim(x, y):
    im_neg = []
    te_neg = []
    for i in range(len(x)):
        im_scores = torch.nn.functional.pairwise_distance(x[i], y)
        im_scores[i] = 0
        im_neg.append(im_scores.max(0)[1])
    for j in range(len(y)):
        te_scores = torch.nn.functional.pairwise_distance(x, y[j])
        te_scores[j] = 0
        te_neg.append(te_scores.max(0)[1])
    return torch.tensor(im_neg), torch.tensor(te_neg)


def calcul_cross_single_inner_loss(score, optim, image_embs, text_embs, target_dist, direct_modal='text'):
    size_v = score.size(0)
    size_t = score.size(1)
    diagonal = score.diag().view(size_v, 1)
    d1 = diagonal.expand_as(score)
    d2 = diagonal.t().expand_as(score)

    # compare every diagonal score to scores in its column
    cost_s = (optim['cross_modal_margin'] + score - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    cost_im = (optim['cross_modal_margin'] + score - d2).clamp(min=0)

    mask = torch.eye(score.size(0)) > .5
    if torch.cuda.is_available():
        mask = mask.cuda()
    cost_s = cost_s.masked_fill_(mask, 0)
    cost_im = cost_im.masked_fill_(mask, 0)

    idx_1, idx_2 = cos_sim(image_embs, text_embs, mask)
    # idx_1, idx_2 = e_sim(image_embs, text_embs)
    # idx_1 = cost_s.max(1)[1]  # the max indexs, [batch_size]
    # idx_2 = cost_im.max(0)[1]
    for i in range(idx_1.shape[0]):
        if idx_1[i] == idx_2[i]:
            idx_2[i] = (idx_1[i] - 1) % idx_1.shape[0]

    # calcul the cross_modal_loss
    # cost_s = cost_s.max(1)[0]
    # cost_im = cost_im.max(0)[0]
    cost_s = cost_s[torch.arange(size_v), idx_1]
    cost_im = cost_im[idx_2, torch.arange(size_t)]
    cross_modal_loss = cost_s.sum() + cost_im.sum()

    if direct_modal == 'text':
        direct_embs = text_embs
    elif direct_modal == 'image':
        direct_embs = image_embs
    else:
        raise ValueError('direct modal must be text or image!')
    with torch.no_grad():
        # dist1_te: [batch_size, 1]
        dist1_te = euclidean_dist(direct_embs, direct_embs[idx_1])  # image_embs is h^A, image_embs[idx_1] is h^A_j
        dist2_te = euclidean_dist(direct_embs, direct_embs[idx_2])  # image_embs[idx_2] is h^A_i , yes

        b = dist1_te > dist2_te
        delta_single = torch.zeros_like(b)
        delta_single[b] = 1
        delta_single = delta_single.float()  # [batch_size, 1]
        delta_single = torch.squeeze(delta_single)  # [batch_size]

    d_i = inner_conduct(target_dist - target_dist[idx_1])
    d_j = inner_conduct(target_dist - target_dist[idx_2])
    diff_cap = d_i - d_j
    diff_cap_norm = torch.sigmoid(diff_cap.clamp(min=-5.0, max=5.0))
    dist_loss1 = (optim['inner_margin'] - delta_single*diff_cap_norm + (1-delta_single)*diff_cap_norm).clamp(min=0)
    some_dist_loss = torch.sum(dist_loss1)

    # diff_cap_norm = diff_cap.clamp(min=-5.0, max=5.0)
    # aa = torch.log(torch.sigmoid(diff_cap_norm))
    # bb = torch.log(1 - torch.sigmoid(diff_cap_norm))
    # some_dist_loss = torch.sum(- (delta_single * aa) + (1 - delta_single) * bb)

    return cross_modal_loss, optim['inner_modal_beta'] * some_dist_loss


def calcul_cross_inner_loss(score, optim, image_embs, text_embs, some_dist):
    size_v = score.size(0)
    size_t = score.size(1)
    diagonal = score.diag().view(size_v, 1)
    d1 = diagonal.expand_as(score)
    d2 = diagonal.t().expand_as(score)

    # compare every diagonal score to scores in its column
    cost_s = (optim['cross_modal_margin'] + score - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    cost_im = (optim['cross_modal_margin'] + score - d2).clamp(min=0)

    mask = torch.eye(score.size(0)) > .5
    if torch.cuda.is_available():
        mask = mask.cuda()
    cost_s = cost_s.masked_fill_(mask, 0)
    cost_im = cost_im.masked_fill_(mask, 0)

    idx_1, idx_2 = cos_sim(image_embs, text_embs, mask)
    # idx_1, idx_2 = e_sim(image_embs, text_embs)

    # calcul the inner modal loss
    # idx_1 = cost_s.max(1)[1]  # the max indexs, [batch_size]
    # idx_2 = cost_im.max(0)[1]

    for i in range(idx_1.shape[0]):
        if idx_1[i] == idx_2[i]:
            idx_2[i] = (idx_1[i] - 1) % idx_1.shape[0]

    # calcul the cross_modal_loss
    # cost_s = cost_s.max(1)[0]
    # cost_im = cost_im.max(0)[0]
    cost_s = cost_s[torch.arange(size_v), idx_1]
    cost_im = cost_im[idx_2, torch.arange(size_t)]

    cross_modal_loss = cost_s.sum() + cost_im.sum()

    with torch.no_grad():
        dist1_im = euclidean_dist(image_embs, image_embs[idx_1])  # image_embs is h^A, image_embs[idx_1] is h^A_j
        dist2_im = euclidean_dist(image_embs, image_embs[idx_2])  # image_embs[idx_2] is h^A_i , yes
        dist1_te = euclidean_dist(text_embs, text_embs[idx_1])
        dist2_te = euclidean_dist(text_embs, text_embs[idx_2])

        a = dist1_im > dist2_im
        b = dist1_te > dist2_te
        a = a.float()
        b = b.float()
        a = a.reshape(a.shape[0])
        b = b.reshape(b.shape[0])
        delta_im = (a + b) / 2
        mask_delta = torch.eq(delta_im, 0.5)

        mask_delta_a = torch.eq(a-b, 1.0)
        mask_delta_b = torch.eq(a-b, -1.0)

        diff_abs_dist = torch.abs(dist1_im - dist2_im) - torch.abs(dist1_te - dist2_te)
        diff_abs_dist = l2norm_r(diff_abs_dist) * 5
        delta_im_sig = torch.sigmoid(diff_abs_dist)
        delta_im_sig = delta_im_sig.reshape(delta_im_sig.shape[0])

        delta_im[mask_delta] = delta_im_sig[mask_delta]
        delta_im[mask_delta_b] = 1 - delta_im[mask_delta_b]

    d_01 = inner_conduct(some_dist - some_dist[idx_1])
    d_02 = inner_conduct(some_dist - some_dist[idx_2])
    diff_cap = d_01 - d_02
    diff_cap_norm = diff_cap.clamp(min=-5.0, max=5.0)

    aa = torch.log(torch.sigmoid(diff_cap_norm))
    bb = torch.log(1 - torch.sigmoid(diff_cap_norm))
    some_dist_loss = torch.sum(- (delta_im * aa) + (1 - delta_im) * bb)

    # print('inner dist loss: {}'.format(some_dist_loss))
    # print('cross modal loss:{} inner dist loss:{}'.format(
    #     cross_modal_loss, some_dist_loss))

    return cross_modal_loss, optim['inner_modal_beta'] * some_dist_loss
    # return some_dist_loss


def calcul_loss(score, labels, optim, img_embs, te_embs, some_dist):
    # score: [batch_size_image, batch_size_text]

    # sample positive and negative samples from the same label data
    # fake_mask = labels[:, 0] > 0.5  # labels is [1, 0]
    # nonfake_mask = labels[:, 1] > 0.5
    # fake_score = score[fake_mask][:, fake_mask]
    # nonfake_score = score[nonfake_mask][:, nonfake_mask]
    # fake_img_embs = img_embs[fake_mask]
    # nonfake_img_embs = img_embs[nonfake_mask]
    # fake_te_embs = te_embs[fake_mask]
    # nonfake_te_embs = te_embs[nonfake_mask]
    # fake_dist = some_dist[fake_mask]
    # nonfake_dist = some_dist[nonfake_mask]
    #
    # # deal with only nonfake_samples or only fake_samples in a batch
    # fake_loss = 0.0
    # nonfake_loss = 0.0
    # fi_loss = 0.0
    # nfi_loss = 0.0
    # if fake_score.numel() > 0:
    #     fake_loss, fi_loss = calcul_cross_inner_loss(
    #                             fake_score, optim, fake_img_embs, fake_te_embs, fake_dist)
    # if nonfake_score.numel() > 0:
    #     nonfake_loss, nfi_loss = calcul_cross_inner_loss(
    #                             nonfake_score, optim, nonfake_img_embs, nonfake_te_embs, nonfake_dist)
    # return fake_loss+nonfake_loss, fi_loss+nfi_loss

    # sample positive and negative samples from the whole data
    cross_loss, inner_loss = calcul_cross_inner_loss(score, optim, img_embs, te_embs, some_dist)

    # unidirectional inner loss, text guide image
    # cross_loss, inner_loss = calcul_cross_single_inner_loss(score, optim, img_embs, te_embs, some_dist, 'image')

    return cross_loss, inner_loss


# only calculate the cross_modal_loss
def calcul_cross_loss(score, optim, image_embs, text_embs):
    size_v = score.size(0)
    size_t = score.size(1)
    diagonal = score.diag().view(size_v, 1)
    d1 = diagonal.expand_as(score)
    d2 = diagonal.t().expand_as(score)

    # compare every diagonal score to scores in its column
    cost_s = (optim['cross_margin'] + score - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    cost_im = (optim['cross_margin'] + score - d2).clamp(min=0)

    mask = torch.eye(score.size(0)) > .5
    if torch.cuda.is_available():
        mask = mask.cuda()
    cost_s = cost_s.masked_fill_(mask, 0)
    cost_im = cost_im.masked_fill_(mask, 0)

    # idx_1 = cost_s.max(1)[1]  # the max indexs, [batch_size]
    # idx_2 = cost_im.max(0)[1]
    # idx_1, idx_2 = cos_sim(image_embs, text_embs, mask)
    # idx_1, idx_2 = e_sim(image_embs, text_embs)

    cost_s = cost_s.max(1)[0]
    cost_im = cost_im.max(0)[0]
    # cost_s = cost_s[torch.arange(size_v), idx_1]
    # cost_im = cost_im[idx_2, torch.arange(size_t)]

    return (cost_s.sum()+cost_im.sum()) * optim['cross_loss_lambda']


# only calculate the inner loss
def calcul_inner_loss(direct_embs, target_embs, optim, target_dist):
    cosine_matrix = torch.nn.functional.cosine_similarity(direct_embs[:, None, :], direct_embs[None, :, :], dim=2)
    mask = torch.eye(cosine_matrix.size(0)) > .5
    if torch.cuda.is_available():
        mask = mask.cuda()
    cosine_matrix = cosine_matrix.masked_fill_(mask, 0)
    idx_1 = cosine_matrix.max(dim=1)[1]
    idx_2 = cosine_matrix.min(dim=1)[1]

    # 为什么不能直接指导单模态表示的cosine相似度呢？而要指导一个距离函数？
    # pos_cos = torch.nn.functional.cosine_similarity(target_embs, target_embs[idx_1])
    # neg_cos = torch.nn.functional.cosine_similarity(target_embs, target_embs[idx_2])
    # inner_loss = (optim['inner_margin'] - pos_cos + neg_cos).clamp(min=0)

    with torch.no_grad():
        # dist1_te: [batch_size, 1]
        dist1_te = euclidean_dist(direct_embs, direct_embs[idx_1])  # image_embs is h^A, image_embs[idx_1] is h^A_j
        dist2_te = euclidean_dist(direct_embs, direct_embs[idx_2])  # image_embs[idx_2] is h^A_i , yes

        b = (dist1_te <= dist2_te)
        delta_single = torch.ones_like(b)
        delta_single[b] = -1
        delta_single = delta_single.float()  # [batch_size, 1]
        delta_single = torch.squeeze(delta_single)  # [batch_size]

    d_i = inner_conduct(target_dist - target_dist[idx_1])
    d_j = inner_conduct(target_dist - target_dist[idx_2])
    diff_cap = d_i - d_j
    diff_cap_norm = torch.sigmoid(diff_cap.clamp(min=-5.0, max=5.0))
    dist_loss1 = (optim['inner_margin'] + delta_single * diff_cap_norm).clamp(min=0)
    # dist_loss1 = (optim['inner_margin'] - delta_single*diff_cap_norm + (1-delta_single)*diff_cap_norm).clamp(min=0)
    return torch.sum(dist_loss1) * optim['inner_loss_beta']


def calcul_class_loss(mm_embs, labels, optim):
    batch_size = mm_embs.size(0)
    class_sim = torch.nn.functional.cosine_similarity(mm_embs[:, None, :], mm_embs[None, :, :], dim=2)
    class_sim = (class_sim + 1) * 0.5  # normalize : 0 < class_sim < 1
    mask = torch.eye(class_sim.size(0)) > .5
    if torch.cuda.is_available():
        mask = mask.cuda()
    class_sim = class_sim.masked_fill_(mask, 0)
    # labels = labels[:batch_size]
    single_labels = labels[:, 0] - labels[:, 1]
    label_matrix = single_labels[:, None] * single_labels[None, :]  # 相同标签的记为1，不同标签的记为-1
    # max是正样本中最大的，min是负样本中最大的的负值
    class_sim = class_sim * label_matrix.float()
    class_loss = (- class_sim.max(1)[0] - class_sim.min(1)[0] + optim['class_margin']).clamp(min=0)
    return class_loss.sum() * optim['class_loss_lambda']


def acc_retrieval(score_matrix):
    image_size = score_matrix.shape[0]
    ranks = np.zeros(image_size)
    top1 = np.zeros(image_size)

    for im_index in range(image_size):
        sort_index = np.argsort(score_matrix[im_index])[::-1]
        rank = 1e20
        te_index = np.where(sort_index == im_index)[0][0]
        if te_index < rank:
            rank = te_index
        if rank == 1e20:
            raise ValueError('rank is 1e20 !!!!!!!!!!')
        ranks[im_index] = rank
        top1[im_index] = sort_index[0]

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return r1, r5, r10, medr, meanr


def retrieval_part(score_matrix, sample_num):
    image_size = score_matrix.shape[0]
    ranks = np.zeros(image_size)
    # top1 = np.zeros(image_size)

    for im_index in range(image_size):
        # 从每个图片要检索的所有文本中，随机采样sample_num个文本数据，记录index，后面的score_matrix只算这几个index数据的
        # 如果随机采样的index里面直接有该图片对应的正文本，记录这个正样本的新位置im_new_index
        # 如果没有，将正文本的原始index随机插入sample_index中，并记录位置im_new_index
        sample_index = np.random.choice(image_size, sample_num)
        if im_index in sample_index:
            im_new_index = np.where(sample_index == im_index)[0][0]
        else:
            im_new_index = np.random.randint(sample_num)
            sample_index[im_new_index] = im_index

        sample_data = score_matrix[im_index, sample_index]
        sort_index = np.argsort(sample_data)[::-1]
        rank = 1e20
        te_index = np.where(sort_index == im_new_index)[0][0]
        if te_index < rank:
            rank = te_index
        if rank == 1e20:
            raise ValueError('rank is 1e20 !!!!!!!!!!')
        ranks[im_index] = rank
        # top1[im_index] = sort_index[0]  # 在这种计算方式下，这个值没有意义了，这里sort_index[0]不再是原始文本的index了．

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return r1, r5, r10, medr, meanr
