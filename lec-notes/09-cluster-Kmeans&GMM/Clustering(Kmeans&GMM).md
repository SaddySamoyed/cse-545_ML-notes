# EM Algorithm in general

## Latent Variable

潜变量（Latent Variables）

- 对于数据 $\mathbf{X}$ 的系统，有时候当我们引入从 $\mathbf{X}$ 中派生出的额外变量 $\mathbf{Z}$，可能更容易理解；但这些变量是未观测的（latent）

  latent variable 的定义是建模的一部分. 

  例如，在高斯混合模型中：对于一个样本 $\mathbf{x}_n$，潜变量 $\mathbf{z}_n$ 表示这个样本属于哪一个高斯分布；

  后验概率 $p(\mathbf{z} \mid \mathbf{x})$ 被称为 **responsibility**，即 latent variable 在给定原数据下的 distribution.



notation:$\mathbf{X}$ 表示所有观测数据，第 $n$ 行为 $\mathbf{x}_n^\top$， $\mathbf{Z}$ 表示所有潜变量，第 $n$ 行为 $\mathbf{z}_n^\top$



$\implies$

如果我们拥有完整数据$\{\mathbf{X}, \mathbf{Z}\}$， 那么我们可以直接

- 当只有 $\mathbf{X}$ 而没有 $\mathbf{Z}$ 时，我们需要对 $\mathbf{Z}$ **边际化（marginalize）**；
  - 这导致对数似然中包含 $\log \sum_z p(x, z)$，使优化变得困难。

- 如果我们拥有完整数据 $\{\mathbf{X}, \mathbf{Z}\}$，就可以直接最大化完全数据似然函数。



EM 算法：总体思想

- **EM（Expectation-Maximization）算法** 是一种通用的优化潜变量模型对数似然的策略。

- 目标是最大化对数似然 $\log p(\mathbf{X} \mid \theta)$：

  - EM 首先引入一个变分分布 $q(\mathbf{Z})$；
  - 构造对数似然的一个下界 $\mathcal{L}(q, \theta)$；
  - 然后交替优化 $q$ 和 $\theta$，直到收敛（类似坐标上升）。

---

# 对数似然的变分下界（Evidence Lower Bound, ELBO）

- 我们的目标是最大化：

  $$
  \log p(\mathbf{X} \mid \theta)
  $$

- 对于任何 $q(\mathbf{Z})$ 都有：

  $$
  \log p(\mathbf{X} \mid \theta) = \mathcal{L}(q, \theta) + \mathrm{KL}(q(\mathbf{Z}) \parallel p(\mathbf{Z} \mid \mathbf{X}, \theta))
  $$

  其中：

  - $\mathcal{L}(q, \theta) = \mathbb{E}_q[\log p(\mathbf{X}, \mathbf{Z} \mid \theta)] - \mathbb{E}_q[\log q(\mathbf{Z})]$
  - KL 散度总是非负，且当且仅当 $q = p(\mathbf{Z} \mid \mathbf{X}, \theta)$ 时取等号

---

# KL 散度与 Jensen 不等式

- KL 散度定义衡量两个分布之间的差异

- 可由 Jensen 不等式推导其非负性：

  $$
  \mathrm{KL}(q \parallel p) = \sum_z q(z) \log \frac{q(z)}{p(z)} \geq 0
  $$

  - 当 $q = p$ 时，KL 散度为 0

---

# EM 算法结构概览

- 重复以下步骤直到收敛：

  ### E 步（期望步）：
  - 固定参数 $\theta$，令 $q(\mathbf{Z}) = p(\mathbf{Z} \mid \mathbf{X}, \theta)$ 使下界最大化

  ### M 步（最大化步）：
  - 固定 $q(\mathbf{Z})$，最大化 $\mathbb{E}_q[\log p(\mathbf{X}, \mathbf{Z} \mid \theta)]$ 得到新的 $\theta$

- 即交替优化 $q$ 和 $\theta$，提升 ELBO 下界，直至收敛

---

# E 步可视化说明

- 对固定的 $\theta$，最大化 ELBO 的最佳 $q$ 是：

  $$
  q(\mathbf{Z}) = p(\mathbf{Z} \mid \mathbf{X}, \theta)
  $$

- 所以 E 步就是计算当前参数下的后验概率

---

# M 步可视化说明

- 固定 $q$，我们最大化：

  $$
  \mathbb{E}_q[\log p(\mathbf{X}, \mathbf{Z} \mid \theta)]
  $$

- 此期望是对完全数据似然的加权平均，M 步更新参数以增加该值

---

# EM 算法总结

1. 随机初始化参数 $\theta$
2. 重复以下步骤直到收敛：
   - **E 步**：设 $q(\mathbf{Z}) = p(\mathbf{Z} \mid \mathbf{X}, \theta)$
   - **M 步**：更新参数 $\theta$ 以最大化期望对数似然
3. 当后验分布不可得时，使用变分分布近似 $q(\mathbf{Z})$

---

是否需要我将这个翻译后的版本导出为 `.md` 文件，或者连同你前面上传的 K-means / GMM 幻灯片内容整合成一个完整的课程笔记？可以告诉我你的偏好！











# K-means

无监督学习（**Unsupervised Learning**）是机器学习的一种类型，它的核心特点是：**训练数据没有标签**。也就是说，算法在学习时，并不知道哪些是“对”或者“错”的答案，它只能从数据中自己“找规律”

Clustering 问题是一类 unsupervised learning 问题.

example: customer segmentation. 将客户分组，以便在决策（如信用卡审批）或营销（如产品推广）中提供帮助。



k-means Algorithm 的 idea 即: 

- 给定 **无标签** 数据  
  $\{ \mathbf{x}^{(n)} \} \quad (n = 1, \ldots, N)$
- **假设这些数据属于 $K$ 个 clusters**（ex: $K = 2$）
- 找到这些簇



使用指示变量 $r_{nk} \in \{0, 1\}, 1\leq n \leq N,1\leq k \leq K$ 作为待学习的 parameter：

- $r_{nk} = 1$ 当 $\mathbf{x}^{(n)}$ 属于第 $k$ 个簇时

- $r_{nk} = 0$ otherwise





寻找 cluster center $\mu_k$ 和分配变量 $r_{nk}$，以 minimize **distortion measure $J$**：
$$
J = \sum_{k=1}^K \sum_{n=1}^N r_{nk} \left\| \mathbf{x}^{(n)} - \mu_k \right\|^2
$$

其中 cluster center 的计算公式为：
$$
\mu_k = \frac{1}{N_k} \sum_{n : \mathbf{x}^{(n)} \in \text{cluster } k} \mathbf{x}^{(n)} = \frac{\sum_{n=1}^N r_{nk} \mathbf{x}^{(n)}}{\sum_{n=1}^N r_{nk}}
$$
(当然这是 trivial 的, 就是 mean point)

distortion measure $J$ 就是: squared distance of points from the center of its own cluster, 简称为 intra-cluster variation.



## The K-Means Algorithm

- 初始化簇中心

- 重复以下步骤直到收敛：

  - cluster assignment:（近似于 EM 中的 "E step"）

    将每个点分配给最近的簇中心：
    $$
    r_{nk} =
    \begin{cases}
    1 & \text{若 } k = \arg\min_j \left\| \mathbf{x}^{(n)} - \mu_j \right\|^2 \\
    0 & \text{否则}
    \end{cases}
    $$
    
  - 参数更新：更新簇中心（近似于 EM 中的"M step"）
  
  $$
  \mu_k = \frac{\sum_n r_{nk} \mathbf{x}^{(n)}}{\sum_n r_{nk}}
  $$



这里没有 back 





# Gaussian Mixture 

**K-Means 使用 hard clustering assignment**

- 每个点**只能**属于一个簇。



**高斯混合模型（Mixture of Gaussians）使用 soft clustering**

- 一个点可以被多个簇共同解释。
- 不同的簇会承担不同程度的“责任”（responsibility），即该点属于每个簇的后验概率。
- （实际上，该点只由某一个簇生成，但我们不知道具体是哪个，于是我们为每个簇分配一个概率）



**图示：**如 $(0.97, 0.03)$ 表示该点以 97% 概率属于 簇2，以 3% 概率属于簇2









![{A4E22888-7017-4186-8D9C-B712353A6389}](Clustering(Kmeans&GMM).assets/GMM.png)
