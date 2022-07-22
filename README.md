

## 动机

现在有各种attention/transformer，但是由于实现细节不同，许多方法未必有效，本仓库的目的就是在严格控制变量的情况下测试各种attention的性能，测试任务初步定为：

- 单向语言模型ALM，例如GPT；
- 双向语言模型BLM，例如Bert/RoBerta；
- 视觉分类模型，例如Vit；



## 说明

本仓库的代码基于fairseq，但是功能比fairseq少很多，目标是给出一个更简洁的框架。



## 安装

torch：

```
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

在windows环境，可以能需要如下版本限制：

```
pip install setuptools==52.0.0
```

安装：

```
pip install -e .
```



## 更新日志

- 2022/7/3: 
  - [x] 初始化仓库，完成fairseq代码迁移；
  - [x] 完成fairseq版本lm, mlm测试；
  - [x] 完成trev版本lm测试；
- 2022/7/4~2022/7/10：
  - [x] 完成mlm测试；
  - [x] 完成char level lm测试；
  - [x] 完成vit迁移（基于fairseq或者启一个子项目）；
  - [x] 完成数据预处理测试；
    - [x] enwik8预处理；
    - [x] Wikitext103预处理；
  - [x] 完成代码梳理；
  - [x] 完成fairseq版本和trev版本性能基本对齐；
- 2022/7/11~2022/7/17：
  - [x] 添加速度，内存测试；
  - [x] 完成单头多头测试；
  - [x] 完成norm测试；
- 2022/7/18~2022/7/24：
  - [x] 完成performer, linear transformer测试；
  - [x] 添加训练脚本；
  - [x] 完成速度，内存信息提取脚本；
- 



## To DO

2022/7/18~2022/7/24规划：

- [ ] 尝试加入lra任务；
- [ ] 完成Readme英文版；
- [ ] 完成performer, linear transformer测试；
- [ ] 完成速度，内存信息提取脚本；



## 备注

masked_lm.py是bert

报错：

```
    ret = torch._C._nn.nll_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index)
RuntimeError: Expected object of scalar type Long but got scalar type Int for argument #2 'target' in call to _thnn_nll_loss_forward

```



## 参考资料

- [https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch)
- [https://github.com/lucidrains/x-transformers](https://github.com/lucidrains/x-transformers)
- [https://github.com/facebookresearch/fairseq](https://github.com/facebookresearch/fairseq)
- [https://github.com/OpenNLPLab/Vicinity-Vision-Transformer](https://github.com/OpenNLPLab/Vicinity-Vision-Transformer)
- [https://github.com/whai362/PVT](https://github.com/whai362/PVT)
- [https://github.com/Oldpan/Pytorch-Memory-Utils](https://github.com/Oldpan/Pytorch-Memory-Utils)
- [https://zhuanlan.zhihu.com/p/424512257](https://zhuanlan.zhihu.com/p/424512257)





