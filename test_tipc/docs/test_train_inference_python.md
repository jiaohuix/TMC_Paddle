# Linux端基础训练推理功能测试

Linux端基础训练推理功能测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能，包括裁剪、量化、蒸馏。

## 1. 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称 | 单机单卡 | 单机多卡 | 多机多卡 | 模型压缩（单机多卡） |
| :------: | :------: | :------: | :------: | :------: | :------------------: |
|   TMC    |   TMC    | 正常训练 |    -     |    -     |          -           |


- 推理相关：基于训练是否使用量化，可以将训练产出的模型可以分为`正常模型`和`量化模型`，这两类模型对应的推理功能汇总如下，

| 算法名称 | 模型名称 | 模型类型 | device | batchsize | tensorrt | mkldnn | cpu多线程 |
| :------: | :------: | -------- | :----: | :-------: | :------: | :----: | :-------: |
|   TMC    |   TMC    | 正常模型 |  GPU   |    1/1    |    -     |   -    |     -     |
|   TMC    |   TMC    | 正常模型 |  CPU   |    1/1    |    -     |   -    |     -     |


## 2. 测试流程

### 2.1 准备数据

少量数据在test_tipc/lite_data/tiny_sample.npy下，包含一条测试样本。

### 2.2 准备环境


- 安装PaddlePaddle >= 2.2
- 安装AutoLog（规范化日志输出工具）
    ```
    pip install git+https://hub.fastgit.org/LDOUBLEV/AutoLog
    ```

### 2.3 功能测试

<div align="center">
    <img src="./tipc_train_inference.png" width=800">
</div>

测试方法如下所示，希望测试不同的模型文件，只需更换为自己的参数配置文件，即可完成对应模型的测试。

```bash
bash test_tipc/test_train_inference_python.sh ${your_params_file} lite_train_whole_infer
```

以ConvBERT的`Linux GPU/CPU 基础训练推理测试`为例，命令如下所示。

```bash
bash test_tipc/test_train_inference_python.sh test_tipc/configs/TMC/train_infer_python.txt  lite_train_whole_infer
```

输出结果如下，表示命令运行成功。

```
 Run successfully with command - python train.py --output-dir=./log/tmc/lite_train_whole_infer/norm_train_gpus_0 --epochs=5    !
 Run successfully with command - python predict.py --model-path output/model_best  !
 Run successfully with command - python export_model.py --model-path output/model_best --save-inference-dir ./infer_output    !
 Run successfully with command - python deploy/inference_python/infer.py --model-dir ./infer_output --data-path ./test_tipc/lite_data/tiny_sample.npy --use-gpu=True --benchmark=./log/tmc/lite_train_whole_infer/norm_train_gpus_0 --batch-size=1     > ./log/tmc/lite_train_whole_infer/python_infer_gpu_batchsize_1.log 2>&1 
 scores:[1.0000000e+00 1.2581669e-22 1.3909554e-22 1.3290132e-22 1.3519208e-22
 1.3693044e-22 1.7731465e-22 1.2372301e-22 2.2385267e-22 1.3749877e-22]
label_id: 0, prob: 1.0
 Run successfully with command - python deploy/inference_python/infer.py --model-dir ./infer_output --data-path ./test_tipc/lite_data/tiny_sample.npy --use-gpu=False --benchmark=./log/tmc/lite_train_whole_infer/norm_train_gpus_0 --batch-size=1     > ./log/tmc/lite_train_whole_infer/python_infer_cpu_batchsize_1.log 2>&1 !

```



## 3. 更多教程

本文档为功能测试用，更丰富的训练预测使用教程请参考：  

* [模型训练、预测、推理教程](../../README.md)  