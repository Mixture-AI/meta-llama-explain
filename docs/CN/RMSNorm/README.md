# RMSNorm

**RMSNorm** (Root Mean Square Layer Normalization) 是一种针对 Layer Norm 改进的归一化方法。

## What is Layer Norm?

Layer Norm 是 NLP 任务中一种常见的归一化方法，旨在解决深度神经网络中存在的 Internal Covariate Shift 问题，即层与层间的输入分布在训练过程中会不断发生变化。这种现象可能会导致网络训练速度减慢，难以收敛等问题。Layer Norm 在每个样本上统计所有维度的值，计算均值和方差进行归一化操作，使数据在训练过程保持相同的分布，从而减轻 Internal Covariate Shift 问题。

Layer Norm 的计算过程可以公式化表示如下：

$$
\begin{gather}
\bar{a}_i = \cfrac{a_i -\mu}{\sigma}g_i \\
\mu = \cfrac{1}{n} \sum\_{i=1}^{n}a_i \qquad \sigma=\sqrt{\cfrac{1}{n}\sum\_{i=1}^{n}(a_i - \mu)^2}
\end{gather}
$$

其中， $a\in\mathbb{R}^n$ 表示 Layer Norm 进行归一化的向量， $\bar{a}\in\mathbb{R}^n$ 表示经过 Layer Norm 规范后的向量。 $\mu$ 和 $\sigma$ 分别表示输入向量 $a$ 的均值和方差。 $g\in\mathbb{R}^n$ 是一组初始化为 $1$ 的增益系数，用来缩放归一化后的结果，来保证归一化操作不会破坏输入的信息。

## Why RMSNorm？

Layer Norm 成功地应用于不同深度神经网络，帮助稳定网络训练，加速模型收敛。然而，其较大的计算开销也限制了网络的速度。RMSNorm 假设了重中心不变性 (re-centering invariance) 在归一化方法中是没有必要的，从而**设计了一种舍弃重中心化操作的归一化方法来降低计算量**，并通过实验验证了提出的假设。

RMSNorm 在对输入做归一化时只进行缩放操作，缩放系数为均方根，公式可以表示如下：

$$
\begin{gather}
\bar{a}_i = \cfrac{a_i}{\textbf{RMS}(\mathbf{a})}g_i,\qquad \textbf{RMS}(\mathbf{a})=\sqrt{\cfrac{1}{n}\sum\_{i=1}^{n}a\_{i}^{2}}
\end{gather}
$$


其中， $\textbf{RMS}(a)$ 表示输入张量 $a\in\mathbb{R}^n$ 的均方根。相较于 Layer Norm，RMSNorm 无需计算均值，用均方根代替方差时也自然无需减去均值。



## References
- [Paper: Root Mean Square Layer Normalization](https://arxiv.org/pdf/1910.07467.pdf)
