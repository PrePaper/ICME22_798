import torch
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import utils


def train_part(train_load, model, criterion, optimizer_t, epoch, optim, writter):
    model.train()
    # train_indicator = (epoch // optim['train_change_num']) % 2
    train_num = 0
    train_loss = 0.0
    train_cross_loss = 0.0
    train_inner_loss = 0.0
    train_class_loss = 0.0
    train_fake_loss = 0.0
    # ind_str = 'text' if train_indicator == 0 else 'img'
    skl_acc = 0.0

    for step, train_data in enumerate(train_load):

        input_image, input_text, input_labels, _, _ = train_data
        if torch.cuda.is_available():
            input_image = input_image.cuda()
            input_text = input_text.cuda()
        scores, image_emb, text_emb, image_dis, text_dis, mm_emb, fake_det = model(input_image, input_text)
        torch.cuda.synchronize()

        pre_labels = fake_det.clone().detach()
        pre_labels[pre_labels > 0.5] = 1
        pre_labels[pre_labels <= 0.5] = 0
        acc = accuracy_score(input_labels.detach().cpu(), pre_labels.cpu())
        skl_acc += acc * input_image.size(0)

        fake_detection_loss = criterion(fake_det, input_labels.float())
        # class_loss = utils.calcul_class_loss(mm_emb, input_labels, optim) / input_image.size(0)
        cross_loss = utils.calcul_cross_loss(scores, optim, image_emb, text_emb) / input_image.size(0)
        # inner_loss = utils.calcul_inner_loss(image_emb, text_emb, optim, text_dis) / input_image.size(0)
        inner_loss = utils.calcul_inner_loss(text_emb, image_emb, optim, image_dis) / input_image.size(0)
        loss_all = inner_loss + cross_loss + fake_detection_loss
        # loss_all = fake_detection_loss + cross_loss

        optimizer_t.zero_grad()
        loss_all.backward()
        torch.cuda.synchronize()
        optimizer_t.step()
        torch.cuda.synchronize()
        # else:
        #     # inner_loss = utils.calcul_inner_loss(image_emb, text_emb, optim, text_dis)
        #     inner_loss = utils.calcul_inner_loss(text_emb, image_emb, optim, image_dis) / input_image.size(0)
        #     loss_all = inner_loss + cross_loss + fake_detection_loss
        #
        #     optimizer_i.zero_grad()
        #     loss_all.backward()
        #     torch.cuda.synchronize()
        #     optimizer_i.step()
        #     torch.cuda.synchronize()

        train_num += input_image.size(0)
        train_loss += loss_all
        train_fake_loss += fake_detection_loss
        train_cross_loss += cross_loss
        train_inner_loss += inner_loss
        # train_class_loss += class_loss

        # niter = epoch * len(train_load) + step + 1
        # if niter % optim['print_freq'] == 0:
        #     writter.add_scalar('train_loss/cross_loss', cross_loss, global_step=niter)
        #     writter.add_scalar('train_loss/inner_loss', inner_loss, global_step=niter)
        #     writter.add_scalar('train_loss/fake_loss', fake_detection_loss, global_step=niter)
        #     writter.add_scalar('train_loss/class_loss', class_loss, niter)
        #     writter.add_scalar('train_loss/all_loss', loss_all, niter)

    print('epoch {:<3d}:train loss {:<6f} cross {:<6f} inner {:<6f} class {:<6f} fake {:<6f} acc {:<6f}'.format(
        epoch, train_loss/len(train_load), train_cross_loss/len(train_load),
        train_inner_loss/len(train_load), train_class_loss/len(train_load), train_fake_loss/len(train_load),
        skl_acc/train_num), end=' ')


def val_part(val_load, model, criterion, epoch, optim, writter, data_type='val'):
    model.eval()
    val_loss_all = 0.0
    val_cross_loss = 0.0
    val_inner_loss = 0.0
    val_class_loss = 0.0
    val_fake_loss = 0.0
    skl_acc = 0.0
    val_num = 0
    for step, val_data in enumerate(val_load):
        input_image, input_text, input_label, _, _ = val_data
        scores, image_emb, text_emb, image_dis, text_dis, mm_emb, fake_dec = model(input_image, input_text)

        pre_labels = fake_dec.clone().detach()
        pre_labels[pre_labels > 0.5] = 1
        pre_labels[pre_labels <= 0.5] = 0
        acc = accuracy_score(input_label.detach().cpu(), pre_labels.cpu())
        skl_acc += acc * input_image.size(0)

        fake_loss = criterion(fake_dec, input_label.float())
        # class_loss = utils.calcul_class_loss(mm_emb, input_label, optim) / input_image.size(0)
        cross_loss = utils.calcul_cross_loss(scores, optim, image_emb, text_emb) / input_image.size(0)
        inner_loss = utils.calcul_inner_loss(text_emb, image_emb, optim, image_dis) / input_image.size(0)
        # inner_loss = utils.calcul_inner_loss(image_emb, text_emb, optim, text_dis) / input_image.size(0)
        val_loss = inner_loss + fake_loss + cross_loss
        # val_loss = fake_loss + cross_loss

        val_loss_all += val_loss
        val_cross_loss += cross_loss
        val_inner_loss += inner_loss
        # val_class_loss += class_loss
        val_fake_loss += fake_loss
        val_num += input_image.size(0)

        # niter = epoch * len(val_load) + step + 1
        # if niter % optim['print_freq'] == 0:
        #     writter.add_scalar('{}_loss/all_loss'.format(data_type), val_loss, niter)
        #     writter.add_scalar('{}_loss/cross_loss'.format(data_type), cross_loss, niter)
        #     writter.add_scalar('{}_loss/inner_loss'.format(data_type), inner_loss, niter)
        #     writter.add_scalar('{}_loss/fake_loss'.format(data_type), fake_loss, niter)
        #     writter.add_scalar('{}_loss/class_loss'.format(data_type), class_loss, niter)

    print('| {} {:<6f} cross {:<6f} inner {:<6f} class {:<6f} fake {:<6f} acc {:<6f}'.format(
        data_type, val_loss_all/len(val_load), val_cross_loss/len(val_load), val_inner_loss/len(val_load),
        val_class_loss/len(val_load), val_fake_loss/len(val_load), skl_acc/val_num))

    return val_loss_all/len(val_load), skl_acc/val_num


# 测试部分其实和验证部分是一样的，没必要搞两个函数
# 现在有必要了，需要输出分类各个类别的pre, recall, f1值
def test_part(test_load, model):
    print('7.test the best model.............')
    model.eval()
    test_labels = []
    predict_labels = []
    # test_post_ids = []
    for step, test_data in enumerate(test_load):
        input_image, input_text, input_label, _, _ = test_data
        _, _, _, _, _, mm_embs, fake_dec = model(input_image, input_text)
        pre_labels = fake_dec.clone().detach()
        pre_labels[pre_labels > 0.5] = 1
        pre_labels[pre_labels <= 0.5] = 0
        test_labels.extend(input_label.detach().cpu().numpy().tolist())
        predict_labels.extend(pre_labels.cpu().numpy().tolist())
        # test_post_ids.extend(list(input_post_ids))

        # np.concatenate((test_labels, input_label.detach().cpu().numpy()), axis=0)
        # np.concatenate((predict_labels, pre_labels.cpu().numpy()), axis=0)

    test_labels = np.array(test_labels)
    predict_labels = np.array(predict_labels)
    test_acc = accuracy_score(test_labels, predict_labels)
    pres, recalls, f1s, supports = precision_recall_fscore_support(
        test_labels[:, 0], predict_labels[:, 0], labels=[0, 1], average=None)

    # for case study, only count the rumors
    # predict_error_ids = []
    # for i, j, k in zip(test_labels, predict_labels, test_post_ids):
    #     if (i == np.array([0, 1])).all():
    #         if (j == np.array([1, 0])).all():
    #             predict_error_ids.append(k)
    # with open('./mmfake_pre_rumor_to_real_ids_Twitter.txt', 'w') as f2:
    #     f2.write('\n'.join(i for i in predict_error_ids))
    # print('predict rumors wrong number is :', len(predict_error_ids))
    print('test acc is:', test_acc)
    print('fake news results:', pres[0], recalls[0], f1s[0], supports[0])
    print('real news results:', pres[1], recalls[1], f1s[1], supports[1])
    return test_acc, pres[0], recalls[0], f1s[0], supports[0], pres[1], recalls[1], f1s[1], supports[1]


def data_tsne(model, data_loader, data_type, save_dir):
    model.eval()
    img_embs, txt_embs, img_dis, txt_dis = [], [], [], []
    data_embs = []
    data_labels = []
    data_event_labels = []
    for datas in data_loader:
        input_v, input_t, input_l, _, event_labels = datas
        _, image_emb, text_emb, image_dist, text_dist, mm_embs, _ = model(input_v, input_t)
        img_embs.extend(image_emb.detach().cpu().numpy())
        txt_embs.extend(text_emb.detach().cpu().numpy())
        img_dis.extend(image_dist.detach().cpu().numpy())
        txt_dis.extend(text_dist.detach().cpu().numpy())

        data_embs.extend(mm_embs.detach().cpu().numpy())
        data_labels.extend(input_l.detach().cpu().numpy())
        data_event_labels.extend(event_labels.detach().cpu().numpy())
    print('8.save {:<5} embs to files...........'.format(data_type))
    data_embs = np.array(data_embs)
    data_labels = np.array(data_labels)
    data_event_labels = np.array(data_event_labels)
    np.save('{}/{}_embs'.format(save_dir, data_type), data_embs)
    np.save('{}/{}_labels'.format(save_dir, data_type), data_labels)
    np.save('{}/{}_event_labels'.format(save_dir, data_type), data_event_labels)

    # np.save('{}/{}_image_embs'.format(save_dir, data_type), img_embs)
    # np.save('{}/{}_text_embs'.format(save_dir, data_type), txt_embs)
    # np.save('{}/{}_image_dist'.format(save_dir, data_type), img_dis)
    # np.save('{}/{}_text_dist'.format(save_dir, data_type), txt_dis)


def retrieval_exm(data_load, model, epoch, batch_size, writter, data_type):
    model.eval()
    print(' ==> {:<5} retrieval exam '.format(data_type), end='|')
    images = []
    texts = []
    # labels = []
    data_num = 0
    for datas in data_load:
        im, te, la = datas
        data_num += len(im)
        images.append(im)
        texts.append(te)
        # labels.append(la)
    scores_matrix = np.zeros((data_num, data_num))

    for i in range(len(images)):
        for j in range(len(texts)):
            sims, _, _, _, _, _, _ = model(images[i], texts[j])
            scores_matrix[batch_size*i:(batch_size*i+min(batch_size, len(images[i]))),
                          batch_size*j:(batch_size*j+min(batch_size, len(texts[j])))] = sims.cpu().data.numpy()
    del images, texts

    # i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr = utils.acc_retrieval(scores_matrix)
    # t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr = utils.acc_retrieval(np.transpose(scores_matrix))

    i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr = utils.retrieval_part(scores_matrix, 200)
    t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr = utils.retrieval_part(np.transpose(scores_matrix), 200)

    print('  {:<5} i2t r1: {:<7f} r5: {:<7f} r10: {:<7f}'.format(data_type, i2t_r1, i2t_r5, i2t_r10), end=' | ')
    print('  {:<5} t2i r1: {:<7f} r5: {:<7f} r10: {:<7f}'.format(data_type, t2i_r1, t2i_r5, t2i_r10))

    writter.add_scalar('{}_retrieval_i2t/r1'.format(data_type), i2t_r1, epoch)
    writter.add_scalar('{}_retrieval_i2t/r5'.format(data_type), i2t_r5, epoch)
    writter.add_scalar('{}_retrieval_i2t/r10'.format(data_type), i2t_r10, epoch)

    writter.add_scalar('{}_retrieval_t2i/r1'.format(data_type), t2i_r1, epoch)
    writter.add_scalar('{}_retrieval_t2i/r5'.format(data_type), t2i_r5, epoch)
    writter.add_scalar('{}_retrieval_t2i/r10'.format(data_type), t2i_r10, epoch)



