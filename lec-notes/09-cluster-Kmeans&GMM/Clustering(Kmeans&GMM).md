# EM Algorithm in general

## Model with Latent Variable

潜变量（Latent Variables）

- 对于数据 $\mathbf{X}$ 的系统，有时候当我们引入从 $\mathbf{X}$ 中派生出的额外变量 $\mathbf{Z}$，可以使得模型具有更强的表达能力 (表示复杂结构) 以及有时候优化会更加简单.；但这些变量可能是未观测的.

  latent variable 的定义是建模的一部分.

  例如，在高斯混合模型中：对于一个样本 $\mathbf{x}_n$，潜变量 $\mathbf{z}_n$ 表示这个样本属于哪一个高斯分布；

  后验概率 $p(\mathbf{z} \mid \mathbf{x})$ 被称为 **responsibility**，即 latent variable 在给定原数据下的 distribution.



通常, 我们想要做一个 model with paramters $\theta$, maximizing the log-likelihood of the observed data:
$$
\theta_{ml} : =   \arg \max_{\theta} \log p(X \mid\theta ) 
$$
这就是普通的 MLE.


而当我们 model with latent variable $Z$ 的时候, 我们 by Bayesian formula, 可以等同于 maximize: 
$$
\log p(X \mid\theta )  = \log  p(X, Z |\theta )   \, \log (Z  |  \theta)
$$

如果我们拥有完整数据$\{\mathbf{X}, \mathbf{Z}\}$， 那么这个 joint prob $\log  p(X, Z |\theta )$ 很容易求得.

但是, 如果我们并不知道 latent variable 呢 ? 比如说, $Z$ 是一个 missing label. 那么
$$
\log p(X \mid\theta ) = \log  \int  p(X, Z |\theta ) \, dZ
$$
Speicially, $Z$ 是 discrete variable 的情况下: 
$$
\log p(X \mid\theta ) =  \log   \sum_Z p(X, Z |\theta )
$$

这个做法称为 $Z$ 的 **marginalization**. 不得不进行 marginalization 使得想要优化这个式子变得困难 (因为 log 里套了积分/求和)



## decomposition of $\log p(X \mid\theta ) =  \log   \sum_Z p(X, Z |\theta )$ into Evidence Lower Bound 和 KL divergence



目标是最大化对数似然 $\log p(\mathbf{X} \mid \theta)$：


$$
\begin{align*}
\log p(\mathbf{X} \mid \theta) 
&= \sum_{\mathbf{Z}} q(\mathbf{Z}) \log p(\mathbf{X} \mid \theta)  \quad \text{(因为} \sum_{\mathbf{Z}} q(\mathbf{Z}) = 1)\\
&= \sum_{\mathbf{Z}} q(\mathbf{Z}) \log \frac{p(\mathbf{X}, \mathbf{Z} \mid \theta)}{p(\mathbf{Z} \mid \mathbf{X}, \theta)} \\
&= \sum_{\mathbf{Z}} q(\mathbf{Z}) \log \frac{p(\mathbf{X}, \mathbf{Z} \mid \theta)}{q(\mathbf{Z})} \cdot \frac{q(\mathbf{Z})}{p(\mathbf{Z} \mid \mathbf{X}, \theta)} \\
&= \sum_{\mathbf{Z}} q(\mathbf{Z}) \log \frac{p(\mathbf{X}, \mathbf{Z} \mid \theta)}{q(\mathbf{Z})} + \sum_{\mathbf{Z}} q(\mathbf{Z}) \log \frac{q(\mathbf{Z})}{p(\mathbf{Z} \mid \mathbf{X}, \theta)}
\end{align*}
$$

我们把这两个成分，分别定义为

1. variational lower bound of $q$ (或称为 Evidence Lower Bound, ELBO)
   $$
   \begin{align}
   \mathcal{L}(q, \theta) &: =\sum_{\mathbf{Z}} q(\mathbf{Z}) \log \frac{p(\mathbf{X}, \mathbf{Z} \mid \theta)}{q(\mathbf{Z})} \\
   &= \mathbb{E}_{Z\sim q(Z)}[\log p(\mathbf{X}, \mathbf{Z} \mid \theta)] - \mathbb{E}_{Z\sim q(Z)}[\log q(\mathbf{Z})]
   \end{align}
   $$
   
2. KL divergence of $q$ w.r.t. $p$

$$
\begin{align}
\mathrm{KL}(q(\mathbf{Z}) \parallel p(\mathbf{Z} \mid \mathbf{X}, \theta)) &:=\sum_{\mathbf{Z}} q(\mathbf{Z}) \log \frac{q(\mathbf{Z})}{p(\mathbf{Z} \mid \mathbf{X}, \theta)}\\
& = \mathbb{E}_{Z\sim q(Z)}[\log q(\mathbf{Z})]  - \mathbb{E}_{Z\sim q(Z)}[\log p(\mathbf{Z}\mid \mathbf{X} ,\theta)] 
\end{align}
$$

从而: 
$$
\begin{aligned} \log p(\mathbf{X} \mid \theta)  & =\mathcal{L}(q, \theta)+K L(q(\mathbf{Z}) \| p(\mathbf{Z} \mid \mathbf{X}, \theta))\end{aligned}
$$


###  KL 散度

我们这里简写.

可以发现, KL-divergence 有很多种展开形式：
$$
\begin{aligned} K L(q \| p) &=\mathbb{E}_{z\sim q(z)}\left[\log \frac{q(z)}{p(z)}\right]\\ 
& =\sum_z q(z) \log \frac{q(z)}{p(z)} \\
& =-\sum_z q(z) \log p(z)+\sum_z q(z) \log q(z)  \\
& = \mathbb{H}(q, p)-\mathbb{H}(q)
\end{aligned}
$$
因而两个分布的 KL 散度就是 $p,q$ 的 cross entropy 减去 $p$ 自己的 entropy.

这个结果是 0 当且仅当 $p=q$ （下面会证明）. 并且显然 $p,q$ 这两个分布越是相近, 这个 KL 散度越低.

但是 notice: 
$$
K L(q \| p) \not =  K L(p \| q)
$$
这个行为不是 commutative 的.

$KL(p‖q)$ 表示：真实分布是 $p$，用 $q$ 来近似它

$KL(q‖p)$ 表示：真实分布是 $q$，用 $p$ 来近似它







### Jensen 不等式: showing that KL 散度是非负的

Recall 凸分析中我们有 Jensen 不等式:

If $f$ is **convex**, then for any $0 \leq \theta_i \leq 1(\forall i)$ s.t. $\theta_1+\theta_2+\cdots+\theta_k=1 \\$ 都有:
$$
\begin{array}{r}
f\left(\theta_1 x_1+\theta_2 x_2+\cdots+\theta_k x_k\right) \leq \theta_1 f\left(x_1\right)+\cdots+\theta_k f\left(x_k\right)
\end{array}
$$
这是 convex ineq 的一个更广泛的 interpolation 推广. 它还可以从而推广到积分形式: 对于任意一个 measure space $(\Omega, A, \mu)$, 如果 measurable $f$ 是 convex 的，那 取任意 measurable function $g$ 都有: 
$$
\varphi\left(\int_{\Omega} f \mathrm{~d} \mu\right) \leq \int_{\Omega} \varphi \circ f \mathrm{~d} \mu
$$
从而 corollary: 
$$
f(\mathbb{E}[x]) \leq \mathbb{E}[f(x)]
$$

因而，由于 log 函数是 convex 的，我们可以由 Jensen 证明出：散度总是非负，且当且仅当 $q = p(\mathbf{Z} \mid \mathbf{X}, \theta)$ 时取等号
$$
\begin{aligned} K L(q \| p) &=\mathbb{E}_{z\sim q(z)}\left[\log \frac{q(z)}{p(z)}\right]\\
&\geq \log (\mathbb{E}_{z\sim q(z)}\left[ \frac{q(z)}{p(z)}\right])    \\
& = -\log (\underbrace{\sum_z q(z) \frac{p(z)}{q(z)}}_{=\sum_z p(z)=1}) \\ & =0\end{aligned}
$$

recall: 
$$
\begin{align*} \log p(\mathbf{X} \mid \theta)  & =\mathcal{L}(q, \theta)+K L(q(\mathbf{Z}) \| p(\mathbf{Z} \mid \mathbf{X}, \theta))\end{align*}
$$
由于 KL 非负, 我们可以得到: 对于任意的 distribution $q$ of $\mathbf{Z}$, $\mathcal{L}(q, \theta)$ 是 $\log p(\mathbf{X})$ 的一个 lower bound. 因此我们才把它叫做 **variational lower bound**. 



## EM Algorithm

EM Algorithm 是一个在设定了 latent variable $Z$ 时，间接地 maximizing (log) $ p(X \mid\theta )$ 的方法. 它并不直接优化 $ p(X \mid\theta )$, 而是通过不断 maximize 它的 variational lower bound $\mathcal{L}(q, \theta)$ 的行为，来间接优化 $ p(X \mid\theta )$ 





重复以下步骤直到收敛：

1. E-step (expectation): 固定参数 $\theta$，compute posterior $p(\mathbf{Z} \mid \mathbf{X}, \theta)$, 然后把它赋给 $q(\mathbf{Z})$，使 variational lower bound $\mathcal{L}$ 最大化 (此时 for 固定的 $\theta$, 有 $\mathcal{L}(q,\theta) = \log p(\mathbf{X} \mid \theta)$)

   具体要做的即:
   $$
   q(\mathbf{Z}) := p(\mathbf{Z} \mid \mathbf{X}, \theta)
   $$
   

2. M-step (maximization)：固定 $q(\mathbf{Z})$，最大化 $\mathbb{E}_q[\log p(\mathbf{X}, \mathbf{Z} \mid \theta)]$ 得到新的 $\theta$

   具体要做的即: 
   $$
   \operatorname{argmax}_\theta \mathcal{L}(q, \theta):=\operatorname{argmax}_\theta \sum_{\mathbf{Z}} q(\mathbf{Z}) \log p(\mathbf{X}, \mathbf{Z} \mid \theta)
   $$

EM 即交替优化 $q$ 和 $\theta$，提升 ELBO 下界，直至收敛





E 步可视化说明

- 对固定的 $\theta$，最大化 ELBO 的最佳 $q$ 是：

  $$
  q(\mathbf{Z}) = p(\mathbf{Z} \mid \mathbf{X}, \theta)
  $$

- 所以 E 步就是计算当前参数下的后验概率



M 步可视化说明

- 固定 $q$，我们最大化：

  $$
  \mathbb{E}_q[\log p(\mathbf{X}, \mathbf{Z} \mid \theta)]
  $$

- 此期望是对完全数据似然的加权平均，M 步更新参数以增加该值





EM 算法总结

1. 随机初始化参数 $\theta$
2. 重复以下步骤直到收敛：
   - **E 步**：设 $q(\mathbf{Z}) = p(\mathbf{Z} \mid \mathbf{X}, \theta)$
   - **M 步**：更新参数 $\theta$ 以最大化期望对数似然
3. 当后验分布不可得时，使用变分分布近似 $q(\mathbf{Z})$













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
