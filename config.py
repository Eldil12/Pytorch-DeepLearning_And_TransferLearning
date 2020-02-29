# coding:utf8
import warnings
import torch as t


class DefaultConfig(object):
    env = 'main'  # visdom环境
    model = 'resnet50'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    # 各类别对应的数字标识
    labels = {'agriculture': 0, 'commercial': 1, 'harbor': 2, 'idle_land': 3, 'industrial': 4, 'meadow': 5}
              #'overpass': 6, 'park': 7, 'pond': 8, 'residential': 9, 'river': 10, 'water': 11}

    # 类别数目
    num_labels = len(labels)

    train_data_root = '.\\data\\train\\'  # 训练集存放路径
    test_data_root = '.\\data\\test\\'  # 测试集存放路径
    load_model_path = './checkpoints/ResNet_0228_15-51-30.pth'#'./checkpoints/ResNet_0224_21-24-45.pth'  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 16  # 每一批的数据量
    use_gpu = True  # 是否使用GPU
    num_workers = 4  # 使用多进程加载的进程数，0代表不使用多进程
    print_freq = 20  # 每训练print_freq个batch则绘制一次曲线

    max_epoch = 30  # 最大训练轮数
    lr = 0.001  # 初始学习率
    lr_decay = 0.5  # 学习率衰减
    weight_decay = 1e-5  # 损失函数

    k = 5  # top-k准确率
    ensemble = True  # 采样决策融合分类

    result_file = 'result.csv'

    def parse(self, kwargs):
        """
        根据字典kwargs更新配置参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has no attribute %s" % k)
            setattr(self, k, v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

        # 打印配置信息
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


opt = DefaultConfig()
