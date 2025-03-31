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

