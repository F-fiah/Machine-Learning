---
cssclasses:
  - 机器学习
tags:
  - 2026/06/1
---
生成学习算法的核心：找到每个class对应的features

# 贝叶斯 Rule

以二分类问题为例：
记 x 为 feature，y 为 class
$$ \begin{align} P(y=1\mid x)&=\frac{P(x\mid y=1)P(y=1)}{P(x)} \\[7pt] 其中 \quad P(x)&=P(x\mid y=1)P(y=1)+P(x\mid y=0)P(y=0) \end{align} $$
# 高斯判别分析(GDA)

假设 $P(x\mid y)$ 服从高斯分布，即固定类别 y 的前提下，任一特征 x 服从多维正态高斯
如果 y 有n和类别，就有n个类型的高斯分布，即同一类别内部特征服从一个（多元）高斯

---
$z \sim \mathcal{N}(\vec{\mu},\Sigma),\quad z\in\mathbb{R}^n,\ \vec{\mu}\in\mathbb{R}^n,\ \Sigma\in\mathbb{R}^{n\times n}$ 
其中 $\mu$ 为均值向量，$\Sigma$ 为协方差方阵，描述特征 i 和特征 j 之间的相关协方差
$$ \begin{align*} \mathbb{E}[z] &= \mu \\ \mathrm{Cov}(z) &= \mathbb{E}\big[(z-\mu)(z-\mu)^\top\big] \\ &= \mathbb{E}[zz^\top] - \big(\mathbb{E}z\big)\big(\mathbb{E}z\big)^\top \end{align*} $$
其中 $Cox(z)$ 为协方差，其期望值即为 $\Sigma$

对于多维的变量 x，高斯函数为：$$ p(x)=\frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac12}} \exp\left\{ -\frac12(x-\mu)^T\Sigma^{-1}(x-\mu) \right\} $$![[多元高斯.png|377]]

---

以二分类问题为例，其基本假设为：
- 同一类别内部特征服从一个（多元）高斯分布
- y本身服从伯努利分布
$$ \begin{align} p(x|y=0)&=\frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^\frac12}\exp\left(-\frac12(x-\mu_0)^T\Sigma^{-1}(x-\mu_0)\right)\\ p(x|y=1)&=\frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^\frac12}\exp\left(-\frac12(x-\mu_1)^T\Sigma^{-1}(x-\mu_1)\right) \\[8pt] p(y) &= \phi^y(1-\phi)^{1-y}\end{align}$$

如何拟合参数：
假设现在有数据集：$\{x_i,y_i\},\quad i = 1,2,3 \dots m$

定义 Joint likelihood（全部m条样本同时发生的联合概率）
$$ \begin{align} \mathcal{L}(\phi,\mu_0,\mu_1,\Sigma)&=\prod_{i=1}^m p(x^{(i)},y^{(i)};\phi,\mu_0,\mu_1,\Sigma)\\ &=\prod_{i=1}^m p(x^{(i)}\mid y^{(i)})\,p(y^{(i)}) \end{align} $$
调整参数使得 L 最大化，可以推导出：
$$ \begin{align} \phi &= \frac{\sum_{i=1}^m y^{(i)}}{m} =\frac{\sum_{i=1}^m \boldsymbol 1\{y^{(i)}=1\}}{m}\\[4pt] \mu_0 &= \frac{\sum_{i=1}^m \boldsymbol 1\{y^{(i)}=0\}x^{(i)}}{\sum_{i=1}^m \boldsymbol 1\{y^{(i)}=0\}}\\[6pt] \mu_1 &= \frac{\sum_{i=1}^m \boldsymbol 1\{y^{(i)}=1\}x^{(i)}}{\sum_{i=1}^m \boldsymbol 1\{y^{(i)}=1\}} \\ \Sigma &= \frac1m\left[ \sum_{y^{(i)}=0}\big(x^{(i)}-\mu_0\big)\big(x^{(i)}-\mu_0\big)^T +\sum_{y^{(i)}=1}\big(x^{(i)}-\mu_1\big)\big(x^{(i)}-\mu_1\big)^T \right] \end{align} $$

将参数调整好后，如何进行prodiction：（现在有的数据只有 $p(x|y), p(y)$ ）
选取y，使得 $P(y|x) = p(y|x)=\dfrac{p(x|y)p(y)}{p(x)}$ 最大，即 $\mathop{\arg\max}_{y}\ p(y|x)$

![[GDA.png|341]]

## Compare GDA to Logistic Congression

GDA是一种特殊的逻辑回归，因为它加强了假设（高斯分布）

进一步推广，如果假设对于每一个y，x服从的是其它的指数家族分布规律，则该算法依然是特殊的逻辑回归

所以当数据的信息不足时，最好使用逻辑回归
同时，当数据很多时，也最好使用逻辑回归，因为数据量大可以弥补数据信息少的缺点
通常先对数据进行统计验证，看数据服从哪种类型的分布

GDA的优势：high efficiency，不需要迭代计算
# 朴素贝叶斯

以 text classification 问题为例

首先将文本数据映射为特征向量，$x[i] = 1\{\text{word appears in my dictionary}\}$

基本假设：
- 在给定标签 y 的条件下，$x_i$ 相互独立，即一个word出现的概率不会影响另一个word的概率 

- $\phi_{j|y=1}=P(x_j=1\mid y=1)$：标签 $y=1$ 时，第 j 个特征 $x_j=1$ 的条件概率
- $\phi_{j|y=0}=P(x_j=1\mid y=0)$：同上
- $\phi_y=P(y=1)$：样本 $y=1$ 的概率

理论计算可得：
$$ \begin{cases} \phi_y=\displaystyle\frac{\sum_{i=1}^m\boldsymbol1\{y^{(i)}=1\}}{m}\\[6pt] \phi_{j|y=k}=\displaystyle\frac{\sum_{i=1}^m\boldsymbol1\{x_j^{(i)}=1,\,y^{(i)}=k\}}{\sum_{i=1}^m\boldsymbol1\{y^{(i)}=k\}},\quad k\in\{0,1\} \end{cases} $$
