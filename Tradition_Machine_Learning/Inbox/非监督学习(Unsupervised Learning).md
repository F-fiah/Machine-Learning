---
cssclasses:
  - 机器学习
tags:
  - 2026/07/05
---
在无监督学习中，训练样本的标记是未知的，需要通过对无标记的训练样本的学习来揭示数据之间的内在性质及其规律
# 聚类（Clustering）
## K-means 聚类算法

```pseudo
\begin{algorithm} \caption{K-Means聚类算法} \begin{algorithmic} \INPUT{样本集 $D = \{x_1, x_2, \dots, x_m\}$；聚类簇数 $k$} \OUTPUT{簇划分结果 $\mathcal{C} = \{C_1,C_2,\dots,C_k\}$} \STATE 从 $D$ 中随机选取 $k$ 个样本作为初始均值向量 $\{\mu_1, \mu_2, \dots, \mu_k\}$ \REPEAT \STATE 令 $C_i = \varnothing \quad (1 \le i \le k)$ \FOR{$j = 1,2,\dots,m$} \STATE 计算样本 $x_j$ 与各均值向量 $\mu_i \ (1 \le i \le k)$ 的距离：$d_{ji} = \|x_j - \mu_i\|_2$ \STATE 根据距离最近的均值向量确定 $x_j$ 的簇标记：$\lambda_j = \arg\min_{i \in \{1,2,\dots,k\}} d_{ji}$ \STATE 将样本 $x_j$ 划入对应簇：$C_{\lambda_j} = C_{\lambda_j} \cup \{x_j\}$ \ENDFOR \FOR{$i = 1,2,\dots,k$} \STATE 计算新均值向量：$\mu_i' = \frac{1}{|C_i|}\sum_{x \in C_i} x$ \IF{$\mu_i' \neq \mu_i$} \STATE 将均值向量 $\mu_i$ 更新为 $\mu_i'$ \ELSE \STATE 保持均值向量不变 \ENDIF \ENDFOR \UNTIL{所有均值向量均无更新} \end{algorithmic} \end{algorithm}
```
K 的值通常手动选取
## 高斯混合聚类

通过概率模型表示聚类

假设观测样本$x^{(i)}$ 由一个不可观测的离散隐变量 $z^{(i)}$ 控制，$z^{(i)}$ 代表第 i 个样本属于第 $z^{(i)}$ 个高斯簇
$$ P\left(x^{(i)}, z^{(i)}\right) = P\left(x^{(i)} \mid z^{(i)}\right) P\left(z^{(i)}\right) $$
若所有样本的簇归属隐变量 $z^{(i)}$ 已知，可直接闭式求解全部 GMM 参数
对数联合似然目标 ：$$ \ell(\phi,\mu,\Sigma) = \sum_{i=1}^m \log p\big(x^{(i)},z^{(i)};\phi,\mu,\Sigma\big) $$ $m$：总样本
混合权重 $\phi_j = \dfrac{\sum_{i=1}^m \mathbb{1}\{z^{(i)}=j\}}{m}$
高斯均值 $\mu_j = \frac{\displaystyle \sum_{i=1}^m \mathbb{1}\{z^{(i)}=j\} x^{(i)}}{\displaystyle \sum_{i=1}^m \mathbb{1}\{z^{(i)}=j\}}$
高斯协方差 $\Sigma_j = \frac{\displaystyle \sum_{i=1}^m \mathbb{1}\{z^{(i)}=j\}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T}{\displaystyle \sum_{i=1}^m \mathbb{1}\{z^{(i)}=j\}}$

但实际上，我们并不知道隐变量 $z^{(i)}$

$$ p_{\mathcal{M}}(x) = \sum_{i=1}^{k} \phi_i \cdot p\left(x \mid \mu_i, \Sigma_i\right), $$
其中 $\phi_i$ 为混合系数，即随机抽一个样本，属于第 i 簇的概率
$$ \begin{aligned} p_{\mathcal{M}}(z_j = i \mid x_j) &= \frac{P(z_j = i) \cdot p_{\mathcal{M}}(x_j \mid z_j = i)}{p_{\mathcal{M}}(x_j)} \\ &= \frac{\phi_i \cdot p\left(x_j \mid \mu_i, \Sigma_i\right)}{\displaystyle\sum_{l=1}^{k} \phi_l \cdot p\left(x_j \mid \mu_l, \Sigma_l\right)}. \end{aligned} $$
则：
$$ \phi_i = \frac{1}{m}\sum_{j=1}^m p_{\mathcal{M}}(z_j = i \mid x_j) $$
$$ \mu_i = \frac{\displaystyle \sum_{j=1}^m p_{\mathcal{M}}(z_j = i \mid x_j) \cdot x_j}{\displaystyle \sum_{j=1}^m p_{\mathcal{M}}(z_j = i \mid x_j)} $$
$$ \Sigma_i = \frac{\displaystyle \sum_{j=1}^m p_{\mathcal{M}}(z_j = i \mid x_j) \cdot (x_j-\mu_i)(x_j-\mu_i)^T}{\displaystyle \sum_{j=1}^m p_{\mathcal{M}}(z_j = i \mid x_j)} $$
不断更新这三个值，最终回收敛，即聚类成功
```pseudo
\begin{algorithm} \caption{高斯混合聚类算法} \begin{algorithmic} \INPUT{样本集 $D = \{x_1,x_2,\dots,x_m\}$；高斯混合成分个数 $k$} \OUTPUT{簇划分 $\mathcal{C} = \{C_1,C_2,\dots,C_k\}$} \STATE 初始化高斯混合分布的模型参数 $\{(\phi_i, \mu_i, \Sigma_i) \mid 1 \le i \le k\}$ \REPEAT \FOR{$j = 1,2,\dots,m$} \STATE 计算 $x_j$ 由各混合成分生成的后验概率：$p_{\mathcal{M}}(z_j = i \mid x_j) \ (1 \le i \le k)$ \ENDFOR \FOR{$i = 1,2,\dots,k$} \STATE 计算新均值向量：$\mu_i'$ \STATE 计算新协方差矩阵：$\Sigma_i'$ \STATE 计算新混合系数：$\phi_i'$ \ENDFOR \STATE 将模型参数 $\{(\phi_i, \mu_i, \Sigma_i) \mid 1 \le i \le k\}$ 更新为 $\{(\phi_i', \mu_i', \Sigma_i') \mid 1 \le i \le k\}$ \UNTIL{满足停止条件} \STATE $C_i = \varnothing \quad (1 \le i \le k)$ \FOR{$j = 1,2,\dots,m$} \STATE 确定 $x_j$ 的簇标记 $\lambda_j$ \STATE 将 $x_j$ 划入相应的簇：$C_{\lambda_j} = C_{\lambda_j} \cup \{x_j\}$ \ENDFOR \end{algorithmic} \end{algorithm}
```
## EM 算法

1. E步（求隐变量后验分布） 
   固定当前模型参数 $\theta$，用贝叶斯公式计算隐变量 $z^{(i)}$ 的后验分布 $Q_i(z^{(i)})$ $$ Q_i(z^{(i)}) := P\big(z^{(i)} \mid x^{(i)};\theta\big) $$
2. M步（最大化下界更新参数）
   更新全局参数 $\theta$$$ \theta := \arg\max_{\theta} \sum_{i}\sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})} $$
