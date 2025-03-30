# 1. [25 points] K-means and GMM for Image Compression

In this problem: 我们将 apply K-means 和 Gaussian Mixture Models (GMM) to lossy image compression, 通过 reducing the number of colors used in the image.

input image files: 

1. `mandrill-large.tiff`, $512 \times 512 = 262144$ pixels size, 24-bit color ($3\times 8$ bits 色彩通道, 8 bits 即表示 0~255, 代表 RGB). 

   因而一张图大小是 $262144 \times 3$ B

2.  `mandrill-small.tiff`: $128 \times 128$ pixels version of `mandrill-large.tiff`. 

## 1.1 K-means 

### (a) (auto) implement k-means

work on `keans_gmm.ipynb` 和 `kmean.py`

Treat 每个 pixel 的 $(r,g,b)\in \mathbb{R}^3$, implement 并 run k-means with 16 clusters, on `mandrill-small.tiff`, running 50 updates steps.

Initial centriods 在 `initial_centrids` 里, 因而结果是 deterministic 的.

我们将 implement a general version of K-means algorithm in `kmeans.train_kmeans()`, which will be graded with the provided sample data and some random data. In order to get full points, your implementation should be efficient and fast enough (otherwise you will get only partial points).

**Hint:** You may use `sklearn.metrics.pairwise_distances` function to **compute the distance** between centroids and data points, although it would not be difficult to implement this function (in a vectorized version) on your own.



### (b) test on `mandrill-large.tiff`

After training, 我们在 `mandrill-large.tiff` 上进行 test, replace 每个 pixel 的 $(r,g,b)$ with the value of the closest cluster centroid.

Use the notebook’s plotting code to display the original and compressed images side-by-side, and attach the plots to the write-up. (Note: you should have reasonable image quality/resolution to make the difference discernible). Also, measure and write down the **mean pixel error** between the original and compressed image.



 





### (c) explain: what factor have we compressed

If we represent the image with these reduced 16 colors, by (approximately) what factor have we compressed the image (in terms of bits used to represent pixels) in terms of the data size? Include an explanation of why.





------

## 1.2 Gaussian Mixtures

现在让我们使用高斯混合模型（Gaussian Mixtures，带完整协方差矩阵）来重复前面的图像压缩任务，这次将聚类数设置为 $K = 5$，而不是使用 K-means。

------

### (d) （10 分）(自动评分)

在笔记本 `kmeans_gmm.ipynb` 中实现 GMM 的 EM 算法。

你需要实现 `gmm.train_gmm()` 来训练一个 GMM 模型，该模型将在提供的样例数据以及一些随机数据上进行评分。为了获得满分，你的实现应当足够高效（否则将只能获得部分分数）。

**提示 1：** 可以使用 `scipy.stats.multivariate_normal()` 来计算数据的对数似然（log-likelihood）。

**提示 2：** 在计算 $\gamma(z_{nk})$ 时可以使用 `scipy.special.logsumexp()`。由于 division by small probabilities 可能出现数值不稳定问题，实际中推荐使用对数似然来表示（可能很小的）概率 $N(x^{(n)} \mid \mu_k, \Sigma_k)$，即：
$$
N(x^{(n)} \mid \mu_k, \Sigma_k) = \exp[\log N(x^{(n)} \mid \mu_k, \Sigma_k)]
$$
**注意：** 不要使用（也不需要使用）`scipy` 或 `scikit-learn` 的其他 API。

------

### (e) （3 分）

在训练图像 `mandrill-small.tiff` 上使用 $K = 5$ 的高斯混合模型进行训练。使用提供的初始参数：

- initial mean（`initial_mu`）
- covariance matrices（`initial_sigma`）
- prior distribution of latent cluster（`initial_pi`）

(因而结果是 deterministic 的.)

report:

- log-likelihood of training data after running 50 EM steps
- GMM 模型的参数 $\{(\pi_k, \mu_k) \mid k = 1, ..., 5\}$

无需写出协方差矩阵 $\Sigma_k$。你可以选择：write down the values, or attach visualization plots，其图例中展示了 $\pi_k$ 和 $\mu_k$ 的值

------

### (f) （2 分）

在使用 `mandrill-small.tiff` 训练后，读取测试图像 `mandrill-large.tiff`，并将每个像素的 $(r, g, b)$ 值替换为其 value of latent cluster mean ($\mu_k$). 这里对于每个 pixel, 我们使用 MAP（最大后验估计，Maximum A Posteriori）for the latent cluster-assignment variable.

使用 notebook 中提供的绘图代码，显示原始图像和压缩图像的并排图，并将图像附在报告中

> 注：图像质量应当足够好，以使压缩前后图像的差异可以被分辨

另外，计算并写出 original image 和 compressed image 之间的平均像素误差（mean pixel error）





# 2. [20 分] EM for GDA with missing labels

在本题中，我们将使用 EM 算法来处理标签不完整时的高斯判别分析 (GDA) 问题

假设你有一个数据集，其中一部分数据 is labeled，另一部分 unlabeled. 

我们希望在这个**partially labelled dataset**上学习一个**generative model**. 

> 注：这种学习设定被称为**半监督学习（semi-supervised learning）**，是机器学习中的一个重要研究方向。

In particular, 我们假设我们有 $l$ 个带标签的样本和 $u$ 个未标记的样本，即：
$$
D = \{(x^{(1)}, y^{(1)}), \cdots, (x^{(l)}, y^{(l)}), x^{(l+1)}, \cdots, x^{(l+u)}\}
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





















