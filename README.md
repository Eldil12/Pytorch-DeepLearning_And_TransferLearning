# 项目使用指南


本项目是一个场景分类的初步实现，有待进一步完善......

## 数据
- 把训练集和测试集分别放在`data`目录下的`train`和`test`文件夹中
- `train`和`test`文件夹下有以各类别为名的子文件夹，子文件夹中存放对应类别的图片，如./data/train/agriculture/0001.jpg


## 训练
必须首先启动visdom（一个可视化工具）：

```
python -m visdom.server
```

然后在`config.py`中根据自己的数据类别修改`labels`，使用如下命令启动训练：（方括号中为可选参数，若不填则使用默认配置）

```
python main.py train [--use-gpu=True --env=main --batch-size=32 ...]
```


## 测试
在`config.py`中或通过如下命令的可选参数修改`load_model_path`的值确定模型加载路径，然后进行测试：

```
python main.py test [--use-gpu=False --batch-size=32 ...]
```
