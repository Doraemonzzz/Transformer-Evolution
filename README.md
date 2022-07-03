

## 动机

现在有各种attention/transformer，但是由于实现细节不同，许多方法未必有效，本仓库的目的就是在严格控制变量的情况下测试各种attention的性能，测试任务初步定为：

- 单向语言模型ALM，例如GPT；
- 双向语言模型BLM，例如Bert/RoBerta；
- 视觉分类模型，例如Vit；



## 说明

本仓库的代码基于fairseq，但是功能比fairseq少很多，目标是给出一个更简洁的框架。



## 安装

```
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install setuptools==52.0.0
```



## 更新日志

- 2022/7/3: 
  - 初始化仓库，完成fairseq代码迁移，完成fairseq版本lm, mlm测试；
  - 完成trev版本lm测试；
- 



## To DO

2022/7/4~2022/7/11规划：

- 完成mlm测试；
- 完成char level lm测试；
- 完成vit迁移（基于fairseq或者启一个子项目）；
- 完成数据预处理测试；
- 完成fairseq版本和trev版本性能基本对齐；



## 已完成

- 



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





