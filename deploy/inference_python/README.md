# Inference 推理


# 目录

- [1. 简介]()
- [2. 推理过程]()
    - [2.1 准备推理环境]()
    - [2.2 模型动转静导出]()
    - [2.3 模型推理]()
- [3. FAQ]()


## 1. 简介

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用MKLDNN、CUDNN、TensorRT进行预测加速，从而实现更优的推理性能。

本文档主要基于Paddle Inference的TMC(TRUSTED MULTI-VIEW CLASSIFICATION)模型推理。

更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/inference_cn.html)。


## 2. 推理过程

### 2.1 准备推理环境

安装好PaddlePaddle即可体验Paddle Inference部署能力。


### 2.2 模型动转静导出

使用下面的命令完成`ConvBERT`模型的动转静导出。

```bash
python export_model.py --model_path=./outputs/model_best/ --save_inference_dir=./infer_output
```

最终在`infer_output/`文件夹下会生成下面的3个文件。

```
convbert_infer
     |----inference.pdiparams     : 模型参数文件
     |----inference.pdmodel       : 模型结构文件
     |----inference.pdiparams.info: 模型参数信息文件
```

### 2.3 模型推理


```bash
python deploy/inference_python/infer.py --model_dir=./infer_output/
```

对于下面的文本进行预测

`the problem , it is with most of these things , is the script .`

在终端中输出结果如下。

```
text: the problem , it is with most of these things , is the script ., label_id: 0, prob: 0.9959195256233215
```

表示预测的类别ID是`0`，置信度为`0.9959`，该结果与基于训练引擎的结果完全一致。


## 3. FAQ