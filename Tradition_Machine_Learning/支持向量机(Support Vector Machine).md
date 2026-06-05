---
cssclasses:
  - 机器学习
tags:
  - 2026/06/05
---
# Optimal margin classifier

一种线性的决策边界算法
$$ h_{\boldsymbol w,b}(x)=g\left(\boldsymbol w^\mathrm{T}x+b\right) $$
其中 $x \in \mathbb{R}^n, \quad b \in \mathbb{R}$ （线性编译器）

如何评价一个边界的好坏：
![[SVW.png|337]]
间隔越大，划分效果越好，因为这样增强了边界的鲁棒性
$$ \begin{cases} \text{If } y^{(i)}=1,\quad want \quad \boldsymbol w^\mathrm{T}x^{(i)}+b \gg 0\\[4pt] \text{If } y^{(i)}=-1,\quad want \quad \boldsymbol w^\mathrm{T}x^{(i)}+b \ll 0 \end{cases} $$
## Margin

**函数间隔：**
定义函数间隔 $\hat\gamma^{(i)} = y^{(i)}\big(\boldsymbol w^\mathrm{T}\boldsymbol x^{(i)} + b\big) \quad y^{(i)} \in \{+1,-1\}$，当 $\hat\gamma^{(i)} > 0$ 时，说明分类正确，且 $\hat\gamma^{(i)}$ 越大，说明分类效果越好

但如果 w 和 b 都同时扩大 k 倍，$\hat\gamma^{(i)}$ 也会随之扩大 k 倍
所以可以对 w 和 b 进行 normalize，即 $(\boldsymbol w,b)\rightarrow \left(\frac{\boldsymbol w}{\|\boldsymbol w\|},\frac{b}{\|\boldsymbol w\|}\right)$

**几何间隔：**
定义几何间隔：$\gamma^{(i)}=\dfrac{y^{(i)}\big(\boldsymbol w^\mathrm{T}\boldsymbol x^{(i)}+b\big)}{\|\boldsymbol w\|}$
易得：$\gamma^{(i)}=\dfrac{\hat\gamma^{(i)}}{\|\boldsymbol w\|}$

对于一个训练集的几何间隔，通常采用最坏的 example，即 $\gamma=\min_i \gamma^{(i)}$
## Classifier

选取参数 w 和 b，使得几何间隔 $\gamma$ 最大化，即：
$$ \begin{cases} \displaystyle\max_{\boldsymbol w,b,\hat\gamma}\ \gamma=\frac{\hat\gamma}{\|\boldsymbol w\|}\\[8pt] y_i(\boldsymbol w^\mathrm{T}\boldsymbol x_i+b)\ge \hat\gamma,\quad \forall i=1,\dots,m,\ \hat\gamma>0 \end{cases} $$
若 $(w,b)$ 最优，则 $\forall k > 0,\ (kw,kb)$ 同样最优：$\dfrac{k\hat\gamma}{k\|w\|}=\dfrac{\hat\gamma}{\|w\|}=\gamma$
所以可以强制设定：$\hat\gamma = 1$
所以问题变成：
$$ \begin{aligned} \min_{\boldsymbol w,b}\quad & \|\boldsymbol w\|^2 \\ \text{s.t.}\quad & y^{(i)}\big(\boldsymbol w^\mathrm{T}\boldsymbol x^{(i)}+b\big)\ge1 \end{aligned} $$
