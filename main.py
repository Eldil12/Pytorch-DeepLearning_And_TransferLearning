# coding:utf8
from collections import Counter

import numpy

from config import opt
import torch as t
import torch.nn.functional as F
import models
from data.dataset import SceneData
from torch.utils.data import DataLoader
from torchnet import meter
from utils.visualize import Visualizer
from torch.backends import cudnn
import tqdm


def train(**kwargs):
    """
    训练
    """
    # 根据传入的参数更改配置信息
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    cudnn.enabled = True
    cudnn.benchmark = True

    # step1: 配置并加载模型
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    # step2: 加载数据（训练集和交叉验证集）
    train_data = SceneData(opt.train_data_root, opt.labels, train=True)
    val_data = SceneData(opt.train_data_root, opt.labels, train=False)
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size,
                                shuffle=True, num_workers=opt.num_workers)

    # step3: 目标函数和优化器
    criterion = t.nn.CrossEntropyLoss()  # 交叉熵损失函数
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)  # Adam算法
    """
    # 冻结除全连接层外的所有层，只训练最后的全连接层（用于有全连接层模型的finetune）
    for para in list(model.parameters())[:-1]:
        para.requires_grad = False
    optimizer = t.optim.Adam(params=[model.fc.weight, model.fc.bias], lr=opt.lr, weight_decay=opt.weight_decay)  # Adam算法
    """

    # step4: 统计指标：平滑处理之后的损失，还有混淆矩阵
    loss_meter = meter.AverageValueMeter()  # 能够计算所有数的平均值和标准差，用来统计一次训练中损失的平均值
    confusion_matrix = meter.ConfusionMeter(opt.num_labels)
    previous_loss = 1e100

    # 训练
    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        # 每次读出一个batch的数据训练
        for step, (data, label) in tqdm.tqdm(enumerate(train_dataloader), total=len(train_data)):

            train_input = data.to(opt.device)
            label_input = label.to(opt.device)

            optimizer.zero_grad()  # 梯度清零
            score = model(train_input)  # 调用模型
            loss = criterion(score, label_input)  # 计算损失函数
            loss.backward()  # 反向传播
            optimizer.step()  # 优化

            # 更新统计指标及可视化
            loss_meter.add(loss.item())
            confusion_matrix.add(score.detach(), label_input.detach())

            if step % opt.print_freq == opt.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])

        model.save()

        # 计算验证集上的指标及可视化
        val_cm, val_accuracy = val(model, val_dataloader)
        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
            lr=lr))

        # 如果损失不再下降，则降低学习率
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            # 动态修改学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]


@t.no_grad()
def val(model, dataloader):
    """
    验证
    :param model: 模型
    :param dataloader: 验证集数据
    :return: 混淆矩阵，准确率
    """
    # 模型设置为评估模式
    model.eval()

    confusion_matrix = meter.ConfusionMeter(opt.num_labels)
    for step, (data, label) in enumerate(dataloader):
        val_input = data.to(opt.device)
        score = model(val_input)  # 调用模型
        confusion_matrix.add(score.detach().squeeze(), label.type(t.LongTensor))

    # 模型恢复为训练模式
    model.train()

    # 计算准确率
    cm_value = confusion_matrix.value()
    correct_num = 0
    for i in range(opt.num_labels):
        correct_num += cm_value[i][i]
    accuracy = 100. * correct_num / (cm_value.sum())
    return confusion_matrix, accuracy


@t.no_grad()
def test(**kwargs):
    """
    测试
    """
    # 根据传入的参数更改配置信息
    opt.parse(kwargs)

    # 配置并加载模型，设置为评估模式
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    # 加载测试数据
    test_data = SceneData(opt.test_data_root, opt.labels, test=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    top1 = 0  # Top-1准确率
    topk = 0  # Top-k准确率
    total = test_data.__len__()  # 数据集大小
    results = []

    for ii, (data, img_id, label) in enumerate(test_dataloader):
        # 若使用采样决策融合分类
        if opt.ensemble:
            label_pred, label_probability = sampling_decision_fusion(model, data)
            top1 += t.eq(label.to(opt.device), t.IntTensor(label_pred).to(opt.device)).sum().float().item()
        # 不使用采样决策融合分类
        else:
            test_input = data.to(opt.device)
            score = model(test_input)  # 调用模型
            probability = t.nn.functional.softmax(score, 1).detach()  # 计算各图像对于各类的概率

            # 计算top-1和top-k准确率
            _, maxk = t.topk(score, opt.k, dim=-1)
            label = label.view(-1, 1).to(opt.device)  # 将label从[n]变为[n,1]便于和[n,k]的maxk比较，其中n为batch大小
            top1 += t.eq(label, maxk[:, 0:1].to(opt.device)).sum().float().item()
            topk += t.eq(label, maxk.to(opt.device)).sum().float().item()

            # 根据probability中每一行的最大值及其索引获取分类信息
            label_info = probability.max(dim=1)
            label_pred = label_info[1].detach().tolist()
            label_probability = label_info[0].detach().tolist()

        # 获取分类结果
        batch_results = [[id_.item(), pred_, probability_]
                         for id_, pred_, probability_ in zip(img_id, label_pred, label_probability)]
        results += batch_results

    print('Accuracy of the network on total {} test images: top1={}%; top{}={}%'.format(total, 100 * top1 / total,
                                                                                        opt.k,
                                                                                        100 * topk / total))
    write_csv(results, opt.result_file)

    return results


def write_csv(results, file_name):
    """
    将分类结果写入csv文件
    :param results:分类结果，每一行为图片id、分类结果和分类概率
    :param file_name:写入的文件名
    """
    import csv
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'label_id', 'probability'])
        writer.writerows(results)

        f.close()


def random_scale_stretch(img):
    """
    随机尺度拉伸
    :param img:待处理的图像张量
    :return:拉伸后的图像张量
    """
    k = numpy.random.uniform(0.5, 1.2)  # 尺度变化因子，由均匀分布随机生成

    # 仿射变换
    theta = t.tensor([
        [k, 0, 0],
        [0, k, 0]
    ], dtype=t.float)  # 伸缩变换矩阵
    grid = F.affine_grid(theta.unsqueeze(0), img.unsqueeze(0).size())  # 利用后向变换求出目标图像每个像素在原图像的位置
    output = F.grid_sample(img.unsqueeze(0), grid)  # 重采样函数，采用双线性插值算法

    return output[0]


def sampling_decision_fusion(model, data):
    """
    采样决策融合，从每一图像中随机截取n幅具有随机尺度的影像拉伸至固定尺度输入模型分类，然后统计每一类别出现的次数，将次数最多的类别作为分类结果
    :param model:分类模型
    :param data:一个batch的数据集
    :return:分类结果
    """
    n = 5  # 随机截取图片数

    ensemble_labels = t.empty(data.shape[0], 1).to(opt.device).float()
    img_data = t.empty(data.shape)  # 保存随机尺度拉伸后的数据集

    for i in range(n):
        # 对所有输入数据进行随机尺度拉伸
        for j in range(data.shape[0]):
            img_data[j] = random_scale_stretch(data[j])

        # 调用模型获取输出结果
        test_input = img_data.to(opt.device)
        score = model(test_input)

        # 根据score中每一行的最大值及其索引获取分类信息
        label_info = score.max(dim=1)
        label_pred = label_info[1].detach().view(-1, 1).float()

        # 将每一次的分类结果连接
        ensemble_labels = t.cat([ensemble_labels, label_pred], dim=1)

    ensemble_info = ensemble_labels[:, 1:].tolist()

    # 获取ensemble_info中每一行出现次数最多的类别作为该图像的分类结果存入prediction
    prediction = []
    for labels in ensemble_info:
        prediction.append(Counter(labels).most_common(1)[0][0])

    return prediction, []


if __name__ == '__main__':
    # import fire

    # fire.Fire()
    # train()
    test()
