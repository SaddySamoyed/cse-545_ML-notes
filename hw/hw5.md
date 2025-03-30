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

**提示 2：** 在计算 $\gamma(z_{nk})$ 时可以使用 `scipy.special.logsumexp()`。由于在似然值过小时除法运算可能出现数值不稳定问题，实际中推荐使用对数似然来表示（可能很小的）概率 $N(x^{(n)} \mid \mu_k, \Sigma_k)$，即：
$$
N(x^{(n)} \mid \mu_k, \Sigma_k) = \exp[\log N(x^{(n)} \mid \mu_k, \Sigma_k)]
$$
**注意：** 不要使用（也不需要使用）`scipy` 或 `scikit-learn` 的其他 API。

------

### (e) （3 分）

在训练图像 `mandrill-small.tiff` 上使用 $K = 5$ 的高斯混合模型进行训练。使用提供的初始参数：

- 初始均值（`initial_mu`）
- 协方差矩阵（`initial_sigma`）
- 潜在聚类变量的先验分布（`initial_pi`）

这些参数用于确保结果的确定性。

**写在报告中：**

- 在执行 50 步 EM 后的训练数据的对数似然值；
- GMM 模型的参数 ${(\pi_k, \mu_k) \mid k = 1, ..., 5}$

无需写出协方差矩阵 $\Sigma_k$。你可以选择：

- 直接写出参数值，或
- 附上可视化图，其图例中展示了 $\pi_k$ 和 $\mu_k$ 的值。

------

### (f) （2 分）

在使用 `mandrill-small.tiff` 训练后，读取测试图像 `mandrill-large.tiff`，并将每个像素的 $(r, g, b)$ 值替换为其潜在聚类的均值值（$\mu_k$）。这里使用每个像素的潜在聚类分配变量的 MAP（最大后验估计，Maximum A Posteriori）。

使用笔记本中提供的绘图代码，显示原始图像和压缩图像的并排图，并将图像附在报告中。

> 注：图像质量应当足够好，以使压缩前后图像的差异可以被分辨。

另外，计算并写出原图像和压缩图像之间的平均像素误差（mean pixel error）。









