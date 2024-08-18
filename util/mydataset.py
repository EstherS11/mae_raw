import os
import cv2
import numpy
from albumentations.pytorch.functional import img_to_tensor, mask_to_tensor
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import h5py
import numpy as np
from torch.utils.data import DataLoader

import random


def random_crop(img, label, edge, img_w, img_h):  # 使用cv2.imread返回的是hwc
    height1 = random.randint(0, img.shape[0] - img_h)
    width1 = random.randint(0, img.shape[1] - img_w)
    # print('img shape:', img.shape[0],img.shape[1])
    # print('label shape:', label.shape[0], label.shape[1])

    height2 = height1 + img_h
    width2 = width1 + img_w

    img = img[height1:height2, width1:width2]
    label = label[height1:height2, width1:width2]
    # print('params:',height1, height2, width1,width2)
    edge = edge[height1:height2, width1:width2]

    return img, label, edge


def read_data(root_dir, names_file, resize):
    if not os.path.exists(names_file):
        print(names_file + 'does not exist!')
    images, labels, edges = [], [], []
    file = open(names_file)
    for f in file:
        if 'None' not in f.split(' ')[1]:
            image_path = os.path.join(root_dir, f.split(' ')[0])
            label_path = os.path.join(root_dir, f.split(' ')[1])
            edge_path = os.path.join(root_dir, f.split(' ')[1])
            if not os.path.isfile(image_path):
                print(image_path + 'does not exist!')
            if not os.path.isfile(label_path):
                print(label_path + 'does not exist!')
            if not os.path.isfile(edge_path):
                print(edge_path + 'does not exist!')

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)

            # if label.shape != edge.shape:
            #     print('image_path:', image_path, image.shape)
            #     print('label_path:', label_path, label.shape)
            #     print('edge_path:', edge_path, edge.shape)

        if 'None' in f.split(' ')[1]:
            image_path = os.path.join(root_dir, f.split(' ')[0])
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = np.zeros(shape=(resize, resize))
            edge = np.zeros(shape=(resize, resize))

        images.append(image)
        labels.append(label)
        edges.append(edge)

    return images, labels, edges


class random_flip(object):
    def __call__(self, img, label, edge):
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
            label = cv2.flip(label, 0)
            edge = cv2.flip(edge, 0)
        if random.random() <= 0.5:
            img = cv2.flip(img, 1)
            label = cv2.flip(label, 1)
            edge = cv2.flip(edge, 1)

        return img, label, edge

class MyDataset_low(Dataset):
    def __init__(self, root_dir, names_file, crop_size=256, crop=False, transform=None):
        self.root_dir = root_dir
        self.names_file = names_file
        self.crop_size = crop_size
        images, labels, edges = read_data(root_dir, names_file, resize=crop_size)
        # labels_np = numpy.array(labels)
        self.crop = crop
        self.transform = transform
        if self.crop == True:
            self.images = self.filter(images)
            self.labels = self.filter(labels)
            self.edges = self.filter(edges)
        else:
            self.images = images
            self.labels = labels
            self.edges = edges
        self.normalize = {"mean": [0.485, 0.456, 0.406],
                          "std": [0.229, 0.224, 0.225]}

    def filter(self, imgs):
        return [img for img in imgs if (
                img.shape[0] >= self.crop_size and
                img.shape[1] >= self.crop_size)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.crop == True:
            img, label, edge = random_crop(self.images[idx], self.labels[idx], self.edges[idx], self.crop_size,
                                           self.crop_size)
        else:
            # breakpoint()
            image_path = self.images[idx]
            label_path = self.labels[idx]
            if self.labels[idx] == 'None':
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                label = np.zeros(shape=(self.crop_size, self.crop_size))
            else:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            # edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(image, (self.crop_size, self.crop_size))
            label = cv2.resize(label, (self.crop_size, self.crop_size))
            edge = cv2.resize(label, (self.crop_size, self.crop_size))

        if self.transform is not None:
            img, label, edge = self.transform(img, label, edge)

        # cv2.imwrite('E:\\JPEG\\code\\MVSS-Net-master\\save_out\\data\\'+str(idx)+'img.png', img)
        # cv2.imwrite('E:\\JPEG\\code\\MVSS-Net-master\\save_out\\data\\'+str(idx)+'label.png', label)
        # cv2.imwrite('E:\\JPEG\\code\\MVSS-Net-master\\save_out\\data\\'+str(idx)+'edge.png', edge)

        img = img_to_tensor(img, normalize=self.normalize)
        label = mask_to_tensor(label, num_classes=1, sigmoid=True)
        # edge = cv2.resize(edge, (int(edge.shape[0]/4), int(edge.shape[1]/4)))
        edge = mask_to_tensor(edge, num_classes=1, sigmoid=True)

        return img, label, edge

class MyDataset(Dataset):
    def __init__(self, root_dir, names_file, crop_size=256, crop=False, transform=None):
        self.root_dir = root_dir
        self.names_file = names_file
        self.crop_size = crop_size
        images, labels, edges = read_data(root_dir, names_file, resize=crop_size)
        # labels_np = numpy.array(labels)
        self.crop = crop
        self.transform = transform
        if self.crop == True:
            self.images = self.filter(images)
            self.labels = self.filter(labels)
            self.edges = self.filter(edges)
        else:
            self.images = images
            self.labels = labels
            self.edges = edges
        self.normalize = {"mean": [0.485, 0.456, 0.406],
                          "std": [0.229, 0.224, 0.225]}

    def filter(self, imgs):
        return [img for img in imgs if (
                img.shape[0] >= self.crop_size and
                img.shape[1] >= self.crop_size)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.crop == True:
            img, label, edge = random_crop(self.images[idx], self.labels[idx], self.edges[idx], self.crop_size,
                                           self.crop_size)
        else:
            img = cv2.resize(self.images[idx], (self.crop_size, self.crop_size))
            label = cv2.resize(self.labels[idx], (self.crop_size, self.crop_size))
            edge = cv2.resize(self.edges[idx], (self.crop_size, self.crop_size))

        if self.transform is not None:
            img, label, edge = self.transform(img, label, edge)

        # cv2.imwrite('E:\\JPEG\\code\\MVSS-Net-master\\save_out\\data\\'+str(idx)+'img.png', img)
        # cv2.imwrite('E:\\JPEG\\code\\MVSS-Net-master\\save_out\\data\\'+str(idx)+'label.png', label)
        # cv2.imwrite('E:\\JPEG\\code\\MVSS-Net-master\\save_out\\data\\'+str(idx)+'edge.png', edge)

        img = img_to_tensor(img, normalize=self.normalize)
        label = mask_to_tensor(label, num_classes=1, sigmoid=True)
        # edge = cv2.resize(edge, (int(edge.shape[0]/4), int(edge.shape[1]/4)))
        edge = mask_to_tensor(edge, num_classes=1, sigmoid=True)

        return img, label, edge


class MyDataset_h5(Dataset):
    def __init__(self, root_dir, mode='train', crop_size=256, crop=False, transform=None):
        self.root_dir = root_dir
        self.img_path = os.path.join(self.root_dir, mode, 'tampered.h5')
        self.mask_path = os.path.join(self.root_dir, mode, 'mask.h5')
        self.edge_path = os.path.join(self.root_dir, mode, 'edge.h5')
        img_h5f = h5py.File(self.img_path, 'r')
        # img_h5f = []

        self.keys = list(img_h5f.keys())
        random.shuffle(self.keys)
        img_h5f.close()

        self.img_size = len(self.keys)

        self.crop_size = crop_size

        self.crop = crop
        self.transform = transform

        self.normalize = {"mean": [0.485, 0.456, 0.406],
                          "std": [0.229, 0.224, 0.225]}



    def __len__(self):
        return self.img_size

    def __getitem__(self, idx):
        # img_idx = idx
        img_h5f = h5py.File(self.img_path, 'r')
        mask_h5f = h5py.File(self.mask_path, 'r')
        edge_h5f = h5py.File(self.edge_path, 'r')
        img = np.uint8(img_h5f[str(idx)])
        mask = np.uint8(mask_h5f[str(idx)])
        edge = np.uint8(edge_h5f[str(idx)])

        if self.crop == True:
            img, mask, edge = random_crop(img, mask, edge, self.crop_size, self.crop_size)
        else:
            img = cv2.resize(img,(self.crop_size, self.crop_size))
            mask = cv2.resize(mask,(self.crop_size, self.crop_size))
            edge = cv2.resize(edge,(self.crop_size, self.crop_size))

        if self.transform is not None:
            img, mask, edge = self.transform(img, mask, edge)


        img = img_to_tensor(img, normalize=self.normalize)
        mask = mask_to_tensor(mask, num_classes=1, sigmoid= True)
        edge = mask_to_tensor(edge, num_classes=1, sigmoid= True)

        img_h5f.close()
        mask_h5f.close()
        edge_h5f.close()

        return img, mask



class MyDataset_rf(Dataset):
    def __init__(self, root_dir, names_file, crop_size=256, crop=False, transform=None):
        self.root_dir = root_dir
        self.names_file = names_file
        self.crop_size = crop_size
        # images, labels, edges = read_data(root_dir, names_file, resize=crop_size)
        # labels_np = numpy.array(labels)
        self.crop = crop
        self.transform = transform
        # if self.crop == True:
        #     self.images = self.filter(images)
        #     self.labels = self.filter(labels)
        #     self.edges = self.filter(edges)
        # else:
        #     self.images = images
        #     self.labels = labels
        #     self.edges = edges
        self.normalize = {"mean": [0.485, 0.456, 0.406],
                          "std": [0.229, 0.224, 0.225]}

        self.val_num = 10
        self.is_train = True
        self.file_path = root_dir
        self.file_path_fake = root_dir
        self.image_names = []
        authentic_names, fake_names = self._img_list_retrieve()
        self.image_class = [authentic_names, fake_names]
        for idx, _ in enumerate(self.image_class):
            self.image_names += _


    def _img_list_retrieve(self):
        authentic_names = self.img_retrieve('authentic.txt', 'authentic')
        fake_names = self.img_retrieve('tampered.txt', 'tampered', False)

        return authentic_names, fake_names


    def img_retrieve(self, file_text, file_folder, real=True):
        '''
            Parameters:
                file_text: str, text file for images.
                file_folder: str, images folder.
            Returns:
                the image list.
        '''
        result_list = []

        data_path = self.file_path if real else self.file_path_fake
        val_num = self.val_num * 3 if file_text in ['Youtube', 'Fashifter'] else self.val_num

        data_text = os.path.join(data_path, file_text)
        data_path = os.path.join(data_path)

        file_handler = open(data_text)
        contents = file_handler.readlines()

        if self.is_train:
            contents_lst = contents[:val_num]
        else:
            contents_lst = contents[-val_num:]

        for content in contents_lst:
            image_name = content.split(' ')[0]
            image_name = os.path.join(data_path, image_name)
            result_list.append(image_name)
        file_handler.close()

        ## only truncate the val_num images.
        if len(result_list) < val_num:
            mul_factor = (val_num//len(result_list))+2
            result_list = result_list * mul_factor
        result_list = result_list[-val_num:]

        return result_list

    def load_mask(self, mask_name, real=False):
        '''binarize the mask, given the mask_name.'''
        if real:
            mask = torch.zeros([1, self.crop_size,self.crop_size])
        else:
            mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.crop_size, self.crop_size))
            mask = mask_to_tensor(mask, num_classes=1, sigmoid=True)
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0

        return mask

    def get_image(self, image_name):
        '''transform the image.'''
        image = cv2.imread(image_name)
        image = cv2.resize(image, (self.crop_size, self.crop_size))
        image = img_to_tensor(image, normalize=self.normalize)

        return image

    def get_mask(self, image_name, cls):
        '''given the cls, we return the mask.'''
        # authentic
        if cls in [0]:
            mask = self.load_mask('', real=True)
        else:
            if '.jpg' in image_name:
                mask_name = image_name.replace('tampered', 'mask').replace('.jpg', '_gt.png')
            elif '.tif' in image_name:
                mask_name = image_name.replace('tampered', 'mask').replace('.tif', '_gt.png')
            else:
                mask_name = image_name
                print(image_name)
            mask = self.load_mask(mask_name)

        return mask

    def get_item(self, index):
        '''
            given the index, this function returns the image with the forgery mask
            this function calls get_image, get_mask for the image and mask torch tensor.
        '''
        image_name = self.image_names[index]
        cls = self.get_cls(image_name)
        print(index, image_name)

        image = self.get_image(image_name)
        mask = self.get_mask(image_name, cls)

        return image, mask

    def get_cls(self, image_name):
        '''return the forgery/authentic cls given the image_name.'''
        if 'authentic' in image_name:
            return_cls = 0
        else:
            return_cls = 1

        return return_cls

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        res = self.get_item(idx)
        return res



# if __name__ == '__main__':
#     dataset_train = MyDataset_rf(root_dir=r'D:\mvss_family\MVSS-Net_v2.0\data\Casiav2_all_train', names_file=None,
#                               crop_size=512, crop=False, transform=None)
#     train_loader = DataLoader(dataset=dataset_train, batch_size=4, shuffle=True, num_workers=0, drop_last=False)
#     for data in train_loader:
#         print('1')