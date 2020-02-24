# coding:utf8
import visdom
import time
import numpy as np


class Visualizer(object):
    """
    封装了visdom的基本操作
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        # 横坐标集合，保存内容如（’loss',23），即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot(self, name, y, **kwargs):
        """
        绘制一个点
        :param name: 曲线名称，如"loss"
        :param y: 纵坐标
        :param kwargs: 其他参数
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=np.unicode(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def plot_many(self, d):
        """
        绘制多点
        :param d: 待绘制点的集合
        """
        for k, v in d.items():
            self.plot(k, v)

    def log(self, info, win='log_text'):
        """
        输出相关信息
        """
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)
