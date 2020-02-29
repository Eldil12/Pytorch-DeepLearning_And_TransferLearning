# coding:utf8
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as t


class SceneData(data.Dataset):

    def __init__(self, root, labels, transforms=None, train=True, test=False):
        """
        获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.labels = labels
        self.test = test

        imgs = []  # 所有图片的地址
        # 为便于计算测试结果的准确率，测试数据与训练数据的目录格式相同，如./data/test/agriculture/0001.jpg
        """
        if self.test:
            for img in os.listdir(root):
                imgs.append(os.path.join(root, img))
        else:
            # 遍历所有种类
            for img_label in os.listdir(root):
                label_dir = os.path.join(root, img_label)
                # 遍历各种类下所有图片
                for img in os.listdir(label_dir):
                    imgs.append(os.path.join(label_dir, img))
        """
        # 遍历所有种类
        for img_label in os.listdir(root):
            label_dir = os.path.join(root, img_label)
            # 遍历各种类下所有图片
            for img in os.listdir(label_dir):
                imgs.append(os.path.join(label_dir, img))

        imgs_num = len(imgs)

        # 将图片文件名的数字标识作为key进行排序
        imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('\\')[-1]))

        # 若进行测试，使用所有数据
        if self.test:
            self.imgs = imgs
        # 若进行训练，使用前80%的数据
        elif train:
            self.imgs = imgs[:int(0.8 * imgs_num)]
        # 若进行验证，使用后20%的数据
        else:
            self.imgs = imgs[int(0.8 * imgs_num):]

        # 预处理
        if transforms is None:
            # 对数据进行归一化
            normalize = t.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            # 若是测试集和验证集
            if self.test or not train:
                self.transforms = t.Compose([
                    t.Resize(224),
                    t.ToTensor(),
                    normalize
                ])
            # 若是训练集
            else:
                self.transforms = t.Compose([
                    t.Resize(256),
                    t.RandomResizedCrop(224),
                    t.RandomHorizontalFlip(),
                    t.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        img_data = Image.open(img_path)
        img_data = self.transforms(img_data)

        if self.test:
            # 若是测试数据，将图片文件名的数字标识作为图片id，将图片父目录的类别对应的数字标识作为label
            img_id = int(self.imgs[index].split('.')[-2].split('\\')[-1])
            label = self.labels[img_path.split('\\')[-2]]
            return img_data, img_id, label
        else:
            # 若是训练数据，将图片父目录的类别对应的数字标识作为label
            label = self.labels[img_path.split('\\')[-2]]
            return img_data, label

    def __len__(self):
        """
        数据集大小
        """
        return len(self.imgs)
