# kg-2019-final
2019年百度的三元组抽取比赛（ http://lic2019.ccf.org.cn/kg ），“科学空间队”源码，最终测试集第7名（本来第8名，第4名不愿提交报告而弃权，躺着前进了一名...），F1为0.8807。

基于“CNN + Attenton + 自行设计的标注结构”的信息抽取模型。

标注结构是自己设计的，我看了很多关系抽取的论文，没有发现类似的做法。所以，如果你基于此模型做出后的修改，最终获奖了或者发表paper什么的，烦请注明一下（其实也不是太奢望）

```
@misc{
  jianlin2019bdkgf,
  title={A Hierarchical Relation Extraction Model with Pointer-Tagging Hybrid Structure},
  author={Jianlin Su},
  year={2019},
  publisher={GitHub},
  howpublished={\url{https://github.com/bojone/kg-2019}},
}
```

## 运行
运行前请用<a href="https://github.com/bojone/kg-2019/blob/master/data_trans.py">data_trans.py</a>转换原始数据。

模型详细介绍： https://kexue.fm/archives/6671

## 环境
Python 2.7 + Keras 2.2.4 + Tensorflow 1.8，其中关系最大的应该是Python 2.7了，如果你用Python 3，需要修改几行代码，至于修改哪几行，自己想办法，我不是你的debugger。

欢迎入坑Keras。人生苦短，我用Keras～

## 交流
QQ交流群：67729435，微信群请加机器人微信号spaces_ac_cn
