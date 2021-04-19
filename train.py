# encoding:utf-8
# ------------------------------------------------------------
# ACM MM 2021 Paper ID: 2418
# Writen by "MUGS: Multimodal Rumor Detection by Multigranular Structure Learning"
# ------------------------------------------------------------

import os
import torch
import argparse
import yaml
import sys
import shutil
from tensorboardX import SummaryWriter
from torchviz import make_dot
import torch.nn as nn

from pytorchtools import EarlyStopping
import utils
# import data_prase
import data_prase_plw
import model
import engine
torch.cuda.reset_max_memory_allocated()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--options_file', default='./MMFake_news_options.yaml', type=str,
                        help='yaml file for some experiment settings.')
    parser.add_argument('--log_dir', default='/media/hibird/study/ALL_MODELS/MMFake_news/logs', type=str,
                        help='dir for logs.')
    parser.add_argument('--data_name', default='Twitter',
                        help='dataset name to process (weibo or Twitter).')
    parser.add_argument('--text_pretrained_model', default='Bert',
                        help='text pretrain model name (w2v, Bert or XLNET).')
    parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate of the model.')
    parser.add_argument('--batch_size', default=128, type=int, help='size of a training batch.')
    parser.add_argument('--num_epochs', default=300, type=int, help='number of training epochs.')
    args = parser.parse_args()

    diff_server_dir = '/media/hibird/data/corpus/fake_news'
    if args.data_name == 'Twitter':
        dataset_dir = '{}/image-verification-corpus/mediaeval2016/plw_preprocess/{}_50_768'.format(
            diff_server_dir, args.text_pretrained_model)
        save_model_dir = '/media/hibird/study/ALL_MODELS/MMFake_news/weibo_weights'
    elif args.data_name == 'weibo':
        dataset_dir = '{}/MM17-WeiboRumorSet/PLW_preprocess/{}_50_768'.format(
            diff_server_dir, args.text_pretrained_model)
        save_model_dir = '/media/hibird/study/ALL_MODELS/MMFake_news/twitter_weights'
    else:
        raise ValueError('dataset must be Twitter or weibo!')

    with open(args.options_file, 'r') as f_opt:
        options = yaml.load(f_opt)

    print('1.load {} data with {} text pretrained model...............'.format(
        args.data_name, args.text_pretrained_model))
    # all_data_loader = data_prase.all_data(options['dataset']['dir'], args.batch_size)
    train_loader, val_loader, test_loader = data_prase_plw.get_loaders(
        dataset_dir, options['dataset']['val_rate'], args.batch_size, options['dataset']['val_from'])

    print('2.build the model..........')
    model1 = model.factory(options['model'], cuda=True)
    print('  model has {} parameters.'.format(utils.params_count(model1)))

    # 创建tensorboard监测模型指标
    new_log_dir = args.log_dir + '/lr{}_ep{}_bs{}/'.format(args.learning_rate, args.num_epochs, args.batch_size)
    if not os.path.exists(new_log_dir):
        os.makedirs(new_log_dir)
    else:
        for files in os.listdir(new_log_dir):
            os.remove(new_log_dir + '/' + files)
    sum_writter = SummaryWriter(log_dir=new_log_dir)
    print('  the tensorboard log dir is:', new_log_dir)

    # 绘制模型网络结构图并保存
    # sum_writter.add_graph(model1)
    # model_network_file = './model_network.png'
    # if os.path.exists(model_network_file):
    #     print('  model netwrok graph exists in {}'.format(model_network_file))
    # else:
    #     for i, j, z in train_loader:
    #         show_y = model1(i, j)
    #         break
    #     show_model = make_dot(show_y)
    #     show_path = show_model.render(filename='model_network',
    #                                   directory='./',
    #                                   view=False,
    #                                   format='png')
    #     print('  model network graph been saved to {}'.format(show_path))

    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.BCEWithLogitsLoss().cuda()

    # params_base = list(model1.linear_calculsim.parameters())
    # params_base += list(model1.fusion.linear_v.parameters())
    # params_base += list(model1.fusion.linear_t.parameters())
    # params_base += list(model1.fusion.bn_layer_v.parameters())
    # params_base += list(model1.fusion.bn_layer_t.parameters())
    # params_base += list(model1.fusion.list_linear_hv.parameters())
    # params_base += list(model1.fusion.list_linear_ht.parameters())
    # params_base += list(model1.fusion.fake_ln1.parameters())
    # params_base += list(model1.fusion.fake_ln2.parameters())
    # params_base += list(model1.fusion.fake_ln3.parameters())
    # params_base += list(model1.fusion.fake_last.parameters())
    # params_base += list(model1.fusion.bn_layer1.parameters())
    # params_base += list(model1.fusion.bn_layer2.parameters())
    # params_base += list(model1.fusion.bn_layer3.parameters())
    # params_base += list(model1.fusion.bn_layer4.parameters())

    # params_img = params_base + list(model1.fusion.dist_learning_v.parameters())
    # params_txt = params_base + list(model1.fusion.dist_learning_t.parameters())
    # optimizer_txt = torch.optim.Adam(params_txt, lr=args.learning_rate, weight_decay=1e-4)
    # optimizer_img = torch.optim.Adam(params_img, lr=args.learning_rate, weight_decay=1e-4)
    optimizer_model1 = torch.optim.Adam(model1.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer_model1, step_size=20, gamma=0.5)

    start_epoch = 0
    # if os.path.isfile(options['optim']['resume']):
    #     checkpoint = torch.load(options['optim']['resume'])
    #     start_epoch = checkpoint['epoch']
    #     best_rsum = checkpoint['best_rsum']
    #     model1.load_state_dict(checkpoint['model1'])
    #     print('=> loaded checkpoint {} (epoch {}, best_rsum {})'.format(
    #         options['optim']['resume'], start_epoch, best_rsum
    #     ))
    #     engine.val_part(val_loader, model1, criterion)
    #     sys.exit(0)
    print('3.start training from checkpoint {} ................'.format(start_epoch))

    # best_val_acc = 0
    model_file = '{}/best_lr{}_es_bs{}.pth.tar'.format(
        save_model_dir, args.learning_rate, args.batch_size)
    early_stopping = EarlyStopping(patience=10, verbose=False, path=model_file)
    ifEarlyStop = False
    # best_acc = 0.0
    for epoch in range(start_epoch, args.num_epochs):

        engine.train_part(train_loader, model1, criterion, optimizer_model1,
                          epoch, options['optim'], sum_writter)

        val_loss, val_acc = engine.val_part(val_loader, model1, criterion,
                                            epoch, options['optim'], sum_writter, 'val')
        # scheduler.step(epoch)
        # print('-------{}---------'.format(scheduler.get_lr()))

        # test_loss, test_acc = engine.val_part(test_loader, model1, criterion,
        #                                       epoch, options['optim'], sum_writter, 'test')

        # if test_acc > best_acc:
        #     best_acc = test_acc

        # if epoch % 1 == 0:
        #     engine.retrieval_exm(train_loader, model1, epoch, args.batch_size, sum_writter, 'train')
        #     engine.retrieval_exm(val_loader, model1, epoch, args.batch_size, sum_writter, 'val')
        #     engine.retrieval_exm(test_loader, model1, epoch, args.batch_size, sum_writter, 'test')

        # 使用early stopping方式停止模型，在验证集loss不下降的时候停止模型训练
        early_stopping(val_loss, model1)
        if early_stopping.early_stop:
            ifEarlyStop = True
            print('4.main model early stop in epoch {}........'.format(epoch-10))
            break
    # print('  the best acc is: ', best_acc)
    if not ifEarlyStop:
        torch.save(model1.state_dict(), model_file)
        print('4.main model did not early stop, finish running at {} epoch.....'.format(args.num_epochs))

    # 上面的训练early stop之后的model1其实不是最好的那个模型，保存下来的才是最好的模型．
    test_model1 = model.factory(options['model'], cuda=True)
    test_model1.load_state_dict(torch.load(model_file))
    whole_acc, fake_pre, fake_rec, fake_f1, fake_su, real_pre, real_rec, real_f1, real_su = \
        engine.test_part(test_loader, test_model1)
    print('5. test acc results: ', whole_acc)
    with open('./final_result_plus.txt', 'a+') as f:
        f.write('{}_myTwitter_{}||{}_{}_{}_{}_{}_{}_{}_{}_{}\n'.format(
            args.data_name, args.text_pretrained_model, args.learning_rate,
            whole_acc, fake_pre, fake_rec, fake_f1, fake_su,
            real_pre, real_rec, real_f1, real_su))

    # engine.data_tsne(test_model1, train_loader, 'train', options['optim']['save_model_dir'])
    # engine.data_tsne(test_model1, val_loader, 'val', options['optim']['save_model_dir'])
    # engine.data_tsne(test_model1, test_loader, 'test', options['optim']['save_model_dir'])

    sum_writter.close()


if __name__ == '__main__':
    main()
