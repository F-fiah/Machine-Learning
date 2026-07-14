---
cssclasses:
  - 计算机视觉
  - 深度学习
tags:
  - 2026/07/14
---
- Discriminative Model :
  Learn a probability distribution $p(\text{label} \mid \text{feature})$
  *No way to handle unreasonable imput*
- Generative Model : 
  Learn a probability distribution $p(\text{features})$
  *Model can reject unreasonable inputs by giving them small probability mass*
- Conditional Generative :
  Learn $p(\text{feature} \mid \text{label})$

$$\underbrace{P(x \mid y)}_{\text{Conditional Generative Model}} = \frac{\overbrace{P(y \mid x)}^{\text{Discriminative Model}}}{\underbrace{P(y)}_{\text{Prior over labels}}} \overbrace{P(x)}^{\text{(Unconditional) Generative Model}}$$
# Explicit Density

显式密度模型：可以写出 / 计算 $p(x)$（显式定义概率公式）
## Tractable density

能精确算出完整 $p(x)$，无近似误差
### Autoregressive 自回归模型

假设数据集 $x_1,x_2...x_N$ 全部是从自然界真实图像分布 $p_{\text{real}}(x)$ 里独立采样出来的合法样本

目标：找到一套模型参数，让训练集里所有数据出现的概率最大化（即模型最贴近真实世界分布）
$$ \begin{aligned} W^* &= \mathop{\arg\max}_{W} \prod_{i} p\left(x^{(i)}\right) \\ &= \mathop{\arg\max}_{W} \sum_{i} \log p\left(x^{(i)}\right) \\ &= \mathop{\arg\max}_{W} \sum_{i} \log f\left(x^{(i)}, W\right) \end{aligned} $$
如果数据具有强相关联（如单张图片内部、单句话内部的一串像素 / 单词），则：
$$ \begin{aligned} p(x) &= p(x_1, x_2, x_3, \dots, x_T) \\ &= p(x_1)p(x_2 \mid x_1)p(x_3 \mid x_1, x_2)\cdots \\ &= \prod_{t=1}^{T} p\left(x_t \mid x_1, \dots, x_{t-1}\right) \end{aligned} $$
## Approximate density

无法直接精确计算 $p(x)$，只能得到近似密度
### Non-Variational Autoencoder

A unsupervised model for learning to extract useful features z from inputs x, without labels

一种有效的方式：将图片 encode 成特征向量，再 decode 成图片，让生成的图片尽可能与原图相近

当输入全新的隐变量 z，就可以通过 decoder 生成全新图片

Problem: How to generate a new z ?
Solution: Froce all z to come from a known distribution
### Variational Autoencoders

假设 z 服从一个先验分布，从该分布中随机采样 z
把采样得到的 z 送入解码器，基于条件分布生成完整图像 x

Problem: How can we train the model ?

**If we had a dataset of (x, z) then train a conditional generative model $p(x \mid z)$**
但隐变量无法观测，无法对数据集进行 z 的标记，无法直接训解码器

$p(x) = \int p(x \mid z)\,p(z)\,dz$，所以我们可以利用原始数据 x 进行训练：最大化 $p(x)$ = 让解码器学会从 z 还原清晰原图
*但积分无法计算*
$$ p_\theta(x) = \frac{p_\theta(x \mid z)\,p_\theta(z)}{p_\theta(z \mid x)} $$
- $p_\theta(x \mid z)$：我们需要训练的参数
- $p_\theta(z)$：服从先验分布

如何得到 $p_\theta(z \mid x)$：采用另一个 encode 神经网络 $q_\phi(z \mid x)$ 进行近似，即 $p_\theta(x) \approx \dfrac{p_\theta(x \mid z)\,p(z)}{q_\phi(z \mid x)}$

如何让神经网络输出一个概率分布：
通过神经网络输出的向量，可以求出均值和标准差，假设服从高斯分布即可

但不能使用该近似式进行训练（不是精确等式，无法直接作为损失进行训练）

经过数学推导可得：
$$ \begin{aligned} \log p_\theta(x) &= \underbrace{\mathbb{E}_{z\sim q_\phi(z|x)}\big[\log p(x|z)\big] - D_{KL}\big(q_\phi(z|x) \parallel p(z)\big)}_{\text{ELBO下界}} + \underbrace{D_{KL}\big(q_\phi(z|x) \parallel p_\theta(z|x)\big)}_{>0} \\ &\ge \mathbb{E}_{z\sim q_\phi(z|x)}\big[\log p_\theta(x|z)\big] - D_{KL}\big(q_\phi(z|x) \parallel p(z)\big) \end{aligned} $$

**Train process :**
1. 输入图片 x 进入 encoder，得到隐变量的均值和方差
   $q_\phi(z \mid x) = \mathcal{N}\big(\mu_{z|x},\, \Sigma_{z|x}\big)$
2. 计算 $D_{KL}\big(q_\phi(z|x) \parallel p(z)\big)$
   $D_{KL}(q \parallel p) = \frac{1}{2}\sum_{i=1}^{d}\left( \mu_i^2 + \exp(\log\sigma_i^2) - \log\sigma_i^2 - 1 \right)$，$p(z) = \mathcal{N}\big(0,\, I\big)$
3. 对 z 进行采样
   $z = \epsilon \odot \Sigma_{z|x} + \mu_{z|x},\quad \epsilon \sim \mathcal{N}(0,I)$
4. 将采样的 z 输进 decoder，得到生成图像的均值
   $p_\theta(x \mid z) = \mathcal{N}\big(\mu_{x|z},\, \sigma^2\big)$，其中 $\sigma$ 为原始图像像素本身的标准差
5. 重新计算总损失
   $\mathcal{L} = \mathbb{E}_{z\sim q_\phi(z|x)}\big[\log p_\theta(x|z)\big] - D_{KL}\big(q_\phi(z|x) \parallel p(z)\big)$
6. 梯度反向传播更新权重
