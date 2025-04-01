无监督学习（**Unsupervised Learning**）是机器学习的一种类型，它的核心特点是：**训练数据没有标签**。也就是说，算法在学习时，并不知道哪些是“对”或者“错”的答案，它只能从数据中自己“找规律”

Clustering 问题是一类 unsupervised learning 问题.

example: customer segmentation. 将客户分组，以便在决策（如信用卡审批）或营销（如产品推广）中提供帮助。

我们将介绍两种 clustering 的算法: K-means 以及 GMM

K-means 其实算是 GMM 的一种特殊情况.



# K-means

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

  - cluster assignment:（E step）

    将每个点分配给最近的簇中心：
    $$
    r_{nk} =
    \begin{cases}
    1 & \text{若 } k = \arg\min_j \left\| \mathbf{x}^{(n)} - \mu_j \right\|^2 \\
    0 & \text{否则}
    \end{cases}
    $$

  - 参数更新：更新簇中心（M step）

  $$
  \mu_k = \frac{\sum_n r_{nk} \mathbf{x}^{(n)}}{\sum_n r_{nk}}
  $$





# Gaussian Mixture 





**K-Means 使用 hard clustering assignment**

- 每个点**只能**属于一个簇。



**高斯混合模型（Mixture of Gaussians）使用 soft clustering**

- 一个点可以被多个簇共同解释。
- 不同的簇会承担不同程度的“责任”（responsibility），即该点属于每个簇的后验概率。
- （实际上，该点只由某一个簇生成，但我们不知道具体是哪个，于是我们为每个簇分配一个概率）



**图示：**如 $(0.97, 0.03)$ 表示该点以 97% 概率属于 簇2，以 3% 概率属于簇2

![Screenshot 2025-03-31 at 15.37.07](09(2)-Clustering(Kmeans&GMM).assets/Screenshot 2025-03-31 at 15.37.07.png)

不同的 clusters take different levels of responsibility for a point. 一个点上, responsibility 即 posterior probability.





## Modeling











## GMM Algorithm



> - Initialize parameters randomly $\theta=\left\{\pi_k, \mu_k, \Sigma_k\right\}_{k=1}^K$
>
> - Repeat until convergence (alternating optimization)
>
> - E Step: Given fixed parameters $\theta$, set $q^{(n)}(\mathbf{z})=p\left(\mathbf{z} \mid \mathbf{x}^{(n)}, \theta\right)$
>   $$
>   \gamma\left(z_{n k}\right)=\frac{\pi_k \mathcal{N}\left(\mathbf{x}^{(n)} \mid \mu_k, \Sigma_k\right)}{\sum_{j=1}^K \pi_j \mathcal{N}\left(\mathbf{x}^{(n)} \mid \mu_j, \Sigma_j\right)}=P\left(z_k=1 \mid \mathbf{x}^{(n)}\right)
>   $$
>   
>
> - M Step: Given fixed $q\left(\mathbf{z}^{(n)}\right)^{\prime}$ 's for $\mathbf{x}^{(n)}$ 's (or $\left.\gamma\left(z_{n k}\right)\right)$, update $\theta$ :
>
>   in order to get 
>   $$
>   \theta^{new} : = \arg \max _\theta \sum_n \sum_{\mathbf{z}} q^{(n)}(\mathbf{z}) \log p\left(\mathbf{z}, \mathbf{x}^{(n)} \mid \theta\right)
>   $$
>   We update:
>   $$
>   \begin{aligned}
>   & \pi_k^{\text {new }}=\frac{N_k}{N}=\frac{\sum_n \gamma\left(z_{n k}\right)}{N} \\
>   & \mu_k^{\text {new }}=\frac{1}{N_k} \sum_{n=1}^N \gamma\left(z_{n k}\right) \mathbf{x}^{(n)} \\
>   & \Sigma_k^{\text {new }}=\frac{1}{N_k} \sum_{n=1}^N \gamma\left(z_{n k}\right)\left(\mathbf{x}^{(n)}-\mu_k^{\text {new }}\right)\left(\mathbf{x}^{(n)}-\mu_k^{\text {new }}\right)^{\top}
>   \end{aligned}
>   $$
>



### Proof of M step update rule

我们上面已经给出 parameter 的更新算法，现在我们证明为什么这些 Parameters 是 argmax 的.

首先，我们先简化表达式：
$$
\begin{aligned}
J(\boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma})= & \sum_{n=1}^N \sum_{k=1}^K q^{(n)}\left(\mathbf{z}_k\right) \log p\left(\mathbf{z}_k, \mathbf{x}^{(n)} \mid \pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\right) \\
= & \sum_{n=1}^N \sum_{k=1}^K \gamma\left(\mathbf{z}_{n k}\right)\left(\log \pi_k+\log \frac{1}{(2 \pi)^{m / 2}\left(\operatorname{det} \boldsymbol{\Sigma}_k\right)^{1 / 2}}-\frac{1}{2}\left(\mathbf{x}^{(n)}-\boldsymbol{\mu}_k\right)^{\top} \boldsymbol{\Sigma}_k^{-1}\left(\mathbf{x}^{(n)}-\boldsymbol{\mu}_k\right)\right. \bigg)\\
= & \sum_{n=1}^N \sum_{k=1}^K \gamma\left(\mathbf{z}_{n k}\right) \log \pi_k-\sum_{n=1}^N \sum_{k=1}^K \gamma\left(\mathbf{z}_{n k}\right) \log \left((2 \pi)^{m / 2}\left(\operatorname{det} \boldsymbol{\Sigma}_k\right)^{1 / 2}\right) \\
= & \sum_{n=1}^N \sum_{k=1}^K \gamma\left(\mathbf{z}_{n k}\right) \log \pi_k-\frac{1}{2} \sum_{n=1}^N \sum_{k=1}^K \gamma\left(\mathbf{z}_{n k}\right)\left(\mathbf{x}^{(n)}-\boldsymbol{\mu}_k\right)^{\top} \boldsymbol{\Sigma}_k^{-1}\left(\mathbf{x}^{(n)}-\boldsymbol{\mu}_k\right) \\
& -\frac{1}{2} \sum_{n=1}^N \sum_{k=1}^K \gamma\left(\mathbf{z}_{n k}\right)\left(\mathbf{x}^{(n)}-\boldsymbol{\mu}_k\right)^{\top} \boldsymbol{\Sigma}_k^{-1}\left(\mathbf{x}^{(n)}-\boldsymbol{\mu}_k\right)+\mathrm{const}
\end{aligned}
$$
因而我们 differentiate:

















![{A4E22888-7017-4186-8D9C-B712353A6389}](09(2)-Clustering(Kmeans&GMM).assets/GMM.png)











# 2. [20 分] EM for GDA with missing labels

在本题中，我们将使用 EM 算法来处理标签不完整时的高斯判别分析 (GDA) 问题

假设你有一个数据集，其中一部分数据 is labeled，另一部分 unlabeled. 

我们希望在这个**partially labelled dataset**上学习一个**generative model**. 

> 注：这种学习设定被称为**半监督学习（semi-supervised learning）**，是机器学习中的一个重要研究方向。

In particular, 我们假设我们有 $l$ 个带标签的样本和 $u$ 个未标记的样本，即：
$$
D = \{(x^{(1)}, y^{(1)}), \cdots, (x^{(l)}, y^{(l)}), x^{(l+1)}, \cdots, x^{(l+u)}\} \notag
$$

We also make the following assumptions

- The data is real-valued $M$ 维向量，即 $x \in \mathbb{R}^M$
- 标签 $y \in \{0, 1\}$（即一个二分类问题）
- We model the data following the same assumption as in GDA:

$$
P(x, y) = P(y) P(x \mid y) \tag{1}
$$

$$
P(y = j) =
\begin{cases}
\phi & \text{if } j = 1 \\
1 - \phi & \text{if } j = 0
\end{cases} \tag{2}
$$

$$
P(x \mid y = j) = \mathcal{N}(x; \mu_j, \Sigma_j), \quad j = 0 \text{ or } 1 \tag{3}
$$

where $\phi$ 是伯努利分布的参数（即 $0 \leq \phi \leq 1$），$\mu_j$ 和 $\Sigma_j$ 分别是第 $j$ 类的 class-specific mean 和 covariance matrix. (因为只有两类, 为了简化记号，也可以写成 $\phi_1 = \phi$，$\phi_0 = 1 - \phi$)

Multivariate Gaussian distribution $\mathcal{N}(x; \mu_j, \Sigma_j)$ 定义如下：

$$
p(x \mid y = j; \mu_j, \Sigma_j) = \mathcal{N}(x; \mu_j, \Sigma_j) =
\frac{1}{(2\pi)^{M/2} |\Sigma_j|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_j)^\top \Sigma_j^{-1}(x - \mu_j)\right) \tag{4}
$$

---

由于存在未标记数据，我们的目标是最大化以下混合目标函数：

$$
J = \sum_{i=1}^l \log p(x^{(i)}, y^{(i)}) + \lambda \sum_{i = l+1}^{l+u} \log p(x^{(i)}) \tag{5}
$$

其中，$\lambda$ 是超参数，用于控制 labeled and unlabeled data 的 weight.

由于我们没有 explicitly model $p(x)$ 的形式，我们使用全概率公式将目标函数重写为：

$$
J = \sum_{i=1}^l \log p(x^{(i)}, y^{(i)}) + \lambda \sum_{i = l+1}^{l+u} \log \sum_{j \in \{0,1\}} p(x^{(i)}, y^{(i)} = j) \tag{6}
$$

通过这种方式，unlabeled training examples 使用了与 labeled samples 相同的模型. 我们将使用 EM 算法来优化上述目标函数.

---

在推导解法时，请给出所有必要的推导步骤，并尽量解释推导的过程，使之易于理解。

> **提示：** 你可以使用如下等式：

$$
p(x^{(i)}, y^{(i)}) = \prod_{j \in \{0,1\}} \left[ \frac{\phi_j}{(2\pi)^{M/2} |\Sigma_j|^{1/2}} \exp\left( -\frac{1}{2}(x^{(i)} - \mu_j)^\top \Sigma_j^{-1}(x^{(i)} - \mu_j) \right) \right]^{\mathbb{I}[y^{(i)} = j]} \tag{7}
$$



### (a) lower bound derivation [3 points]

**(a)** 推导 objective $J$ 的 variational lower bound . Specifically, 证明对于任意的概率分布 $q_i(y^{(i)} = j)$，objective 的 lower bound 可以写为：
$$
L(\mu, \Sigma, \phi) = \sum_{i=1}^l \log p(x^{(i)}, y^{(i)}) + \lambda \sum_{i=l+1}^{l+u} \sum_{j \in \{0,1\}} Q_{ij} \log \frac{p(x^{(i)}, y^{(i)} = j)}{Q_{ij}} \tag{8}
$$

其中，$Q_{ij} \triangleq q_i(y^{(i)} = j)$ 是记号简化。

> **提示：** 可以考虑使用 Jensen 不等式来推导。





### (b) E-step [2 points]

Write down the E-step. Specifically, define the distribution $Q_{ij} = q_i(y^{(i)} = j)$



### (c) M-step for $\mu_k$ [6 points]

推导当 $k = 0$ 或 $1$ 时，$\mu_k$ 的 M-step update rule, while holding $Q_i$'s fixed.

并文字解释：从直觉上看，what $\mu_k$ looks like ? in terms of $x^{(i)}$'s (labeled and unlabeled) 以及 pseudo-counts.



### (d) M-step for $\phi$ 

[6 points] 推导 $\phi \in \mathbb{R}$ 的 M-step update rule, while holding $Q_i$'s fixed.

并文字解释：从直觉上看，what $\phi$ looks like ? in terms of $x^{(i)}$'s (labeled and unlabeled) 以及 pseudo-counts.





### (e) M-step for $\Sigma_k$

[3 points] 最后，based on analogy, 写出 $\Sigma_k$ （$k = 0$ 或 $1$）的 M-step update rule。

由于我们知道推导过程与 GDA（高斯判别分析）或 GMM 的 M 步类似，因此你不需要重复完整推导步骤（你已经在之前任务中练习过）。

并文字解释：从直觉上看，what $\Sigma_k$ looks like ? in terms of $x^{(i)}$'s (labeled and unlabeled) 以及 pseudo-counts.



