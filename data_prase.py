import torch.utils.data as data
import numpy as np
import torch
import torch.utils
import math
from sklearn import preprocessing


class PreDataset(data.Dataset):
    """
    Load Prepare datasets, including images and texts
    Possible datasets: Weibo, Twitter
    """
    def __init__(self, data_path, data_split):
        # the read image_embs from file is of type bytes(element wise),
        # I don't know why, but I have to change it to float
        self.images = np.load('{}/{}_image_embed.npy'.format(data_path, data_split))
        self.images = self.images.astype(np.float)

        self.labels = np.load('{}/{}_label.npy'.format(data_path, data_split))

        self.texts_i = np.load('{}/{}_text_embed.npy'.format(data_path, data_split))
        self.length = len(self.labels)
        texts_dim = self.texts_i.shape[1] * self.texts_i.shape[2]
        self.texts = np.reshape(self.texts_i, (self.length, texts_dim))

    def __getitem__(self, index):
        image_embs = torch.tensor(self.images[index]).float()
        text_embs = torch.tensor(self.texts[index]).float()
        labels = torch.tensor(self.labels[index])

        if torch.cuda.is_available():
            image_embs = image_embs.cuda()
            text_embs = text_embs.cuda()
            labels = labels.cuda()
        return image_embs, text_embs, labels

    def __len__(self):
        return self.length


class SplitDataset(data.Dataset):
    def __init__(self, full_data, indexs):
        self.images, self.texts, self.labels = full_data[indexs]
        self.length = len(self.images)

    def __getitem__(self, item):
        return self.images[item], self.texts[item], self.labels[item]

    def __len__(self):
        return self.length


class GetDataset(data.Dataset):
    def __init__(self, data_path):
        train_images = np.load('{}/train_image_embed.npy'.format(data_path))
        train_images = train_images.astype(np.float)
        train_labels = np.load('{}/train_label.npy'.format(data_path))
        train_texts_i = np.load('{}/train_text_embed.npy'.format(data_path))
        train_length = len(train_labels)
        texts_dim = train_texts_i.shape[1] * train_texts_i.shape[2]
        train_texts = np.reshape(train_texts_i, (train_length, texts_dim))

        test_images = np.load('{}/test_image_embed.npy'.format(data_path))
        test_images = test_images.astype(np.float)
        test_labels = np.load('{}/test_label.npy'.format(data_path))
        test_texts_i = np.load('{}/test_text_embed.npy'.format(data_path))
        test_length = len(test_labels)
        test_texts = np.reshape(test_texts_i, (test_length, texts_dim))

        self.images = np.concatenate((train_images, test_images), axis=0)
        self.labels = np.concatenate((train_labels, test_labels), axis=0)
        self.texts = np.concatenate((train_texts, test_texts), axis=0)
        self.length = train_length + test_length

    def __getitem__(self, item):
        image_embs = torch.tensor(self.images[item]).float()
        text_embs = torch.tensor(self.texts[item]).float()
        labels = torch.tensor(self.labels[item])

        if torch.cuda.is_available():
            image_embs = image_embs.cuda()
            text_embs = text_embs.cuda()
            labels = labels.cuda()
        return image_embs, text_embs, labels

    def __len__(self):
        return self.length


def get_loaders(data_path, val_rate, batch_size, val_from='train'):
    if val_from not in ['train', 'test']:
        raise ValueError('val from must be train or test!!')

    original_train_data = PreDataset(data_path, 'train')
    original_test_data = PreDataset(data_path, 'test')
    data_num = len(original_test_data)

    if val_rate <= 0:
        val_data = original_test_data
        train_data = original_train_data
        test_data = original_test_data
        print('  !!! val data is test data !!!')
    else:
        split_data = original_train_data if val_from == 'train' else original_test_data

        split_data_indexs = np.arange(len(split_data))
        val_number = math.floor(data_num * val_rate)
        np.random.seed(3)
        val_indexs = np.random.choice(split_data_indexs, val_number, False)
        rest_indexs = np.delete(split_data_indexs, val_indexs)
        val_data = SplitDataset(split_data, val_indexs)
        rest_data = SplitDataset(split_data, rest_indexs)

        train_data = rest_data if val_from == 'train' else original_train_data
        test_data = rest_data if val_from == 'test' else original_test_data

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True
                                               )
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                             batch_size=batch_size,
                                             shuffle=True
                                             )
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=batch_size,
                                              shuffle=True
                                              )
    print('  train data number:', len(train_data))
    print('  val   data number:', len(val_data))
    print('  test  data number:', len(test_data))
    return train_loader, val_loader, test_loader


def all_data(data_path, batch_size):
    all_data = GetDataset(data_path)

    all_data_loader = torch.utils.data.DataLoader(dataset=all_data,
                                                  batch_size=batch_size,
                                                  shuffle=True
                                                  )
    print('  all data number:', len(all_data))
    return all_data_loader
