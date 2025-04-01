# 1. [25 points] K-means and GMM for Image Compression

In this problem: We will apply K-means and Gaussian Mixture Models (GMM) to lossy image compression, by reducing the number of colors used in the image.

input image files: 

1. `mandrill-large.tiff`, $512 \times 512 = 262144$ pixels size, 24-bit color ($3\times 8$ bits color channels, 8 bits represent 0~255, RGB colors). 

   So the size of each picture is $262144 \times 3$ B

2.  `mandrill-small.tiff`: $128 \times 128$ pixels version of `mandrill-large.tiff`. 

## 1.1 K-means 

### (a) (auto) implement k-means

work on `keans_gmm.ipynb` 和 `kmean.py`

Treat every pixel as $(r,g,b)\in \mathbb{R}^3$, implement and run k-means with 16 clusters, on `mandrill-small.tiff`, running 50 updates steps.

Initial centriods is in `initial_centrids` , so the result is deterministic.

We will implement a general version of K-means algorithm in `kmeans.train_kmeans()`, which will be graded with the provided sample data and some random data. In order to get full points, your implementation should be efficient and fast enough (otherwise you will get only partial points).

**Hint:** You may use `sklearn.metrics.pairwise_distances` function to **compute the distance** between centroids and data points, although it would not be difficult to implement this function (in a vectorized version) on your own.

### (b) test on `mandrill-large.tiff`

After training, we test on `mandrill-large.tiff`, replace $(r,g,b)$ of every pixel with the value of the closest cluster centroid.

Attach the plots original and compressed images side-by-side to the write-up. (Note: you should have reasonable image quality/resolution to make the difference discernible). 

Also, measure and write down the **mean pixel error** between the original and compressed image.

> **Sol**: 
>
> Plots as below
>
> ![image-20250401150832659](hw5.assets/image-20250401150832659.png)
>
> **mean pixel error** as below:
>
> ![Screenshot 2025-04-01 at 15.08.43](hw5.assets/Screenshot 2025-04-01 at 15.08.43.png)



### (c) explain: what factor have we compressed

If we represent the image with these reduced 16 colors, by (approximately) what factor have we compressed the image (in terms of bits used to represent pixels) in terms of the data size? Include an explanation of why.









------

## 1.2 Gaussian Mixtures

现在让我们使用高斯混合模型（Gaussian Mixtures，带完整协方差矩阵）来重复前面的图像压缩任务，这次将聚类数设置为 $K = 5$，

### (d) implement the EM algorithm for GMM

(10 pts) (Autograder) Work on the notebook `kmeans_gmm.ipynb` to implement the EM algorithm for GMM. You will need to implement gmm. train_gmm() to train a GMM model, which will be graded with the provided sample data and some random data. In order to get full points, your implementation should be efficient and fast enough (otherwise you will get only partial points).

[Hint 1: You may use `scipy.stats.multivariate_normal()` to compute the log-likelihood of the data.] 

[Hint 2: You may use `scipy.special.logsumexp()` when computing $\gamma\left(z_{n k}\right)$. You would need trick this because division by small probabilities can become computationally unstable when the likelihood values are too small. In practice, it is recommended to represent (possibly small) probabilities $\mathcal{N}\left(\mathbf{x}^{(n)} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\right)$ with log-likelihood. Note that
$$
N(x^{(n)} \mid \mu_k, \Sigma_k) = \exp[\log N(x^{(n)} \mid \mu_k, \Sigma_k)]  \notag
$$
 ]
Do not use (and you don't need use) any other scipy or scikit-learn APIs.

>**E-step** (compute responsibilities):
>
>For each point $ x_n $, and cluster $ k $:
>
>>
>
>$$
>\gamma(z_{nk}) = \frac{\pi_k \cdot \mathcal{N}(x_n \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \cdot \mathcal{N}(x_n \mid \mu_j, \Sigma_j)} \notag
>$$
>
>Use **log-space trick**:
>
>- compute `log p(x_n | k)` using `scipy.stats.multivariate_normal.logpdf`
>- then compute log-sum-exp over $ k $for normalization (using `scipy.special.logsumexp`)
>- exponentiate to get $ \gamma(z_{nk})$
>
>
>
>**M-step** (update parameters):
>
>Let $ N_k = \sum_n \gamma(z_{nk}) $
>
>- $ \pi_k = \frac{N_k}{N} $
>- $ \mu_k = \frac{1}{N_k} \sum_n \gamma(z_{nk}) x_n $
>- $ \Sigma_k = \frac{1}{N_k} \sum_n \gamma(z_{nk}) (x_n - \mu_k)(x_n - \mu_k)^T $
>



### (e) train GMM on image 

(3 points) Train GMM on `mandrill-small.tiff` using $K = 5$. Provided initial parameters:

- initial mean（`initial_mu`）
- covariance matrices（`initial_sigma`）
- prior distribution of latent cluster（`initial_pi`）

report:

- log-likelihood of training data after running 50 EM steps
- Parameters $\{(\pi_k, \mu_k) \mid k = 1, ..., 5\}$

You do not need to write down $\Sigma_k$. You can choose either write down the values, or attach visualization plots.

> log-likelihood of training data after running 50 EM steps:
>
> <img src="hw5.assets/Screenshot 2025-04-01 at 15.23.40.png" alt="Screenshot 2025-04-01 at 15.23.40" style="zoom: 33%;" />
>
> Parameter values:
>
> <img src="hw5.assets/Screenshot 2025-04-01 at 15.17.43.png" alt="Screenshot 2025-04-01 at 15.17.43" style="zoom:33%;" />
>
> 
>
> Plots: <img src="hw5.assets/image-20250401151824120.png" alt="image-20250401151824120" style="zoom:50%;" />
>
> 



### (f)

(2 pts) After training on the train `image mandrill-small.tiff`, read the test image `mandrill-large.tiff` and replace each pixel's $(r, g, b)$ values with the value of latent cluster mean, where we use the MAP (Maximum A Posteriori) estimation for the latent cluster-assignment variable for each pixel.

Use the notebook's plotting code to display the original and compressed images side-by-side, and attach the plots to the write-up. (Note: you should have reasonable image quality/resolution to make the difference discernable). 

Also, measure and write down the mean pixel error between the original and compressed image.

> **Sol:** 
>
> Plot:
>
> <img src="hw5.assets/image-20250401152454647.png" alt="image-20250401152454647" style="zoom:33%;" />
>
> mean pixel error between the original and compressed image:
>
> <img src="hw5.assets/Screenshot 2025-04-01 at 15.25.23.png" alt="Screenshot 2025-04-01 at 15.25.23" style="zoom:40%;" />



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













# 3. [20 points] PCA and eigenfaces

### (a) derive PCA from "minimizing squared error" viewpoint

(a) ( 8 pts ) In lecture, we derived PCA from the "maximizing variance" viewpoint. In this problem, we will take the "minimizing squared error" viewpoint. Let $K \in\{1, \ldots, D\}$ be arbitrary and let $\mathbf{x}^{(n)} \in \mathbb{R}^D$. Let $\mathcal{U}=\left\{\mathbf{U}=\left[\mathbf{u}_1 \cdots \mathbf{u}_K\right] \in \mathbb{R}^{D \times K} \mid\left\{\mathbf{u}_i\right\}_{i=1}^K\right.$ 's are orthonormal vectors $\}$, where $\mathbf{u}_i$ is the $i$-th column vector of $\mathbf{U}$.
Let's define the objective function for minimizing the distortion error:
$$
\mathcal{J}=\frac{1}{N} \sum_{n=1}^N\left\|\mathbf{x}^{(n)}-\mathbf{U} \mathbf{U}^{\top} \mathbf{x}^{(n)}\right\|^2=\frac{1}{N} \sum_{n=1}^N\left\|\mathbf{x}^{(n)}-\sum_{i=1}^K \mathbf{u}_i \mathbf{u}_i^{\top} \mathbf{x}^{(n)}\right\|^2   \tag{9}
$$

Here, $\mathbf{U U}^{\top} \mathbf{x}^{(n)}$ is called a projection of $\mathbf{x}^{(n)}$ into the subspace spanned by $\mathbf{u}_i$ 's, and we can denote the projection $\widetilde{\mathbf{x}}^{(n)}=\mathbf{U} \mathbf{U}^{\top} \mathbf{x}^{(n)}$ as in the lecture.
Specifically, show that:
$$
\mathcal{J}=\sum_{i=1}^D \lambda_i-\sum_{i=1}^K \mathbf{u}_i^{\top} \mathbf{S} \mathbf{u}_i	\tag{10}
$$

where $\mathbf{S}$ is the data covariance matrix $\mathbf{S}=\frac{1}{N} \sum_{n=1}^N\left(\mathbf{x}^{(n)}-\overline{\mathbf{x}}\right)\left(\mathbf{x}^{(n)}-\overline{\mathbf{x}}\right)^{\top}, \overline{\mathbf{x}}$ is the data mean vector, and $\lambda_1 \geq \ldots \geq \lambda_d$ are the (ordered) eigenvalues of $\mathbf{S}$. Since the first term is a constant, the above equation implies that minimizing the squared error after projection is equivalent to maximizing the variance, as we have shown in the lecture slides.

With further simplification, show that the minimum distortion error corresponds to the sum of the $D-K$ smallest eigenvalues of $\mathbf{S}$, i.e.,
$$
\min _{\mathbf{U} \in \mathcal{U}} \mathcal{J}=\sum_{k=K+1}^D \lambda_k	\tag{11}
$$
and that the $\mathbf{u}_i$ 's that minimize $\mathcal{J}$ are indeed the $K$ eigenvectors of $\mathbf{S}$ corresponding to the (ordered) eigenvalues $\left\{\lambda_i\right\}_{k=1}^K$. After showing Eq.(10), it is okay to use the fact (without proof) that the optimal solution $\mathbf{u}_i$ 's that maximize $\sum_{i=1}^K \mathbf{u}_i^{\top} \mathbf{S} \mathbf{u}_i$ is to pick the top- $K$ eigenvectors of $\mathbf{S}$ (i.e., $\mathbf{u}_1, \ldots, \mathbf{u}_K$ corresponding to the largest $K$ eigenvalues of $\mathbf{S}$ in descending order), as we already have seen in the lecture.

[Hint 1: You may assume that the data is zero-centered, i.e., $\overline{\mathbf{x}}=\frac{1}{N} \sum_{n=1}^N \mathbf{x}^{(n)}=\mathbf{0} \in \mathbb{R}^D$ without loss of generality. Be sure to mention it if you make such an assumption.]

[Hint 2: You can rewrite the objective as $\mathcal{J}=\frac{1}{N}\left\|\mathbf{X}-\mathbf{U U}^{\top} \mathbf{X}\right\|_F^2$, where $\mathbf{X} \in \mathbb{R}^{D \times N}$ is the matrix that stacks all the data points $\left\{\mathbf{x}^{(n)}\right\}_{n=1}^N$ as column vectors, and the $\|\cdot\|_F$ denotes the Frobenious norm:
$$
\|A\|_F=\sqrt{\sum_{i, j}\left(A_{i j}\right)^2}	\tag{12}
$$

and use the fact $\|A\|_F^2=\operatorname{tr}\left(A^{\top} A\right)$ and $\operatorname{tr}(\mathbf{S})=\sum_i \lambda_i$. Also note that there are many possible approaches for proving the claim, so you do not have to use this fact if you take a different approach.]

Now, you will apply PCA to face images. The principal components (eigenvectors) of the face images are called eigenfaces.

### (b) implement PCA

(4 pts) (Autograder) Work on the provided code pca.ipynb and pca.py to implement PCA. Your code will be graded by the correctness on the sample face dataset and some other randomly-generated dataset.



### (c) perform PCA on face images

 (3 pts) By regarding each image as a vector in a high dimensional space, perform PCA on the face images (sort the eigenvalues in descending order). In the write-up, report the eigenvalues corresponding to the first 10 principal components, and plot all the eigenvalues (in sorted order) where x -axis is the index of corresponding principal components and $y$-axis is the eigenvalue. Use $\log$ scale for the $y$-axis.



### (d) plot eigenfaces

 (3 pts) Plot and attach to your write-up: a $2 \times 5$ array of subplots showing the first 10 principal components/eigenvectors ("eigenfaces") (sorted according to the descending eigenvalues) as images, treating the mean of images as the first principal component. Comment on what facial or lighting variations some of the different principal components are capturing (Note: you don't need to comment for all the images. Just pick a few that capture some salient aspects of image).



### (e) calculate: how many principle components are needed to represent 95% total variance

(2 pts) Eigenfaces are a set of bases for all the images in the dataset (every image can be represented as a linear combination of these eigenfaces). Suppose we have $L$ eigenfaces in total. Then we can use the $L$ coefficients (of the bases, i.e. the eigenfaces) to represent an image. Moreover, we can use the first $K(<L)$ eigenfaces to reconstruct the face image approximately (correspondingly use $K$ coefficients to represent the image). In this case, we reduce the dimension of the representation of images from $L$ to $K$. To determine the proper $K$ to use, we will check the percentage of variance that has been preserved (recall that the basic idea of PCA is preserving variance). Specifically, we define total variance
$$
v(K)=\sum_{i=1}^K \lambda_i \notag
$$

where $1 \leq K<L$ and $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_L$ are eigenvalues. Then the percentage of total variance is


$$
\frac{v(K)}{v(L)} \notag
$$


How many principal components are needed to represent $95 \%$ of the total variance? How about $99 \%$ ? What is the percentage of reduction in dimension in each case?







# 4. [10 points] Independent Component Analysis


In this problem, you will implement maximum-likelihood Independent Component Analysis (ICA) for blind audio separation. As we learned in the lecture, the maximum-likelihood ICA minimizes the following loss:
$$
\ell(W)=\sum_{i=1}^N\left(\sum_{j=1}^m \log g^{\prime}\left(w_j^{\top} x^{(i)}\right)+\log |W|\right) \tag{13}
$$

where $N$ is the number of time steps, $m$ is the number of independent sources, $W$ is the transformation matrix representing a concatenation of $w_j$ 's, and $g(s)=1 /\left(1+e^{-s}\right)$ is the sigmoid function. This link has some nice demos of blind audio separation: https://cnl.salk.edu/~tewon/Blind/blind_audio.html.

We provided the starter code `ica.py` and the `data ica_data.dat`, which contains mixed sound signals from multiple microphones. Run the provided notebook `ica.ipynb` to load the data and run your ICA implementation from `ica.py`.

### (a) implement ICA

 (6 points) (Autograder) Implement ICA by filling in the ica.py file.

### (b) report $W$

(4 points) Run your ICA implementation in the ica.ipynb notebook. To make sure your code is correct, you should listen to the resulting unmixed sources. (Some overlap in the sources may be present, but the different sources should be pretty clearly separated.)
Report the $W$ matrix you found and submit the notebook ica.ipynb (along with ica.py) to the autograder. Make sure the audio tracks are audible in the notebook before submitting. You do not need to submit your unmixed sound files (`ica_unmixed_track_X.wav`).









# 5. [25 points] Conditional Variational Autoencoders

In this problem, you will implement a conditional variational autoencoder (CVAE) from [1] and train it on the MNIST dataset.

### (a) derive variational lower bound of a conditional VAE

[5 points] Derive the variational lower bound of a conditional variational autoencoder. Show that:
$$
\begin{aligned}
\log p_\theta(\mathbf{x} \mid \mathbf{y}) & \geq \mathcal{L}(\theta, \phi ; \mathbf{x}, \mathbf{y}) \\
& =\mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x}, \mathbf{y})}\left[\log p_\theta(\mathbf{x} \mid \mathbf{z}, \mathbf{y})\right]-D_{K L}\left(q_\phi(\mathbf{z} \mid \mathbf{x}, \mathbf{y}) \| p_\theta(\mathbf{z} \mid \mathbf{y})\right)	
\end{aligned}\tag{14}
$$

where $\mathbf{x}$ is a binary vector of dimension $d, \mathbf{y}$ is a one-hot vector of dimension $c$ defining a class, $\mathbf{z}$ is a vector of dimension $m$ sampled from the posterior distribution $q_\phi(\mathbf{z} \mid \mathbf{x}, \mathbf{y})$. The posterior distribution is modeled by a neural network of parameters $\phi$. The generative distribution $p_\theta(\mathbf{x} \mid \mathbf{y})$ is modeled by another neural network of parameters $\theta$. Similar to the VAE that we learned in the class, we assume the conditional independence on the componenets of $\mathbf{z}$ : i.e., $q_\phi(\mathbf{z} \mid \mathbf{x}, \mathbf{y})=\prod_{j=1}^m q_\phi\left(z_j \mid \mathbf{x}, \mathbf{y}\right)$, and $p_\theta(\mathbf{z} \mid \mathbf{y})=\prod_{j=1}^m p_\theta\left(z_j \mid \mathbf{y}\right)$.






### (b) Derive the analytical KL-divergence between two Gaussian distributions 

[8 points] Derive the analytical solution to the KL-divergence between two Gaussian distributions $D_{K L}\left(q_\phi(\mathbf{z} \mid \mathbf{x}, \mathbf{y}) \| p_\theta(\mathbf{z} \mid \mathbf{y})\right)$. Let us assume that $p_\theta(\mathbf{z} \mid \mathbf{y}) \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and show that:
$$
D_{K L}\left(q_\phi(\mathbf{z} \mid \mathbf{x}, \mathbf{y}) \| p_\theta(\mathbf{z} \mid \mathbf{y})\right)=-\frac{1}{2} \sum_{j=1}^m\left(1+\log \left(\sigma_j^2\right)-\mu_j^2-\sigma_j^2\right)	\tag{15}
$$

where $\mu_j$ and $\sigma_j$ are the outputs of the neural network that estimates the parameters of the posterior distribution $q_\phi(\mathbf{z} \mid \mathbf{x}, \mathbf{y})$.
You can assume without proof that
$$
D_{K L}\left(q_\phi(\mathbf{z} \mid \mathbf{x}, \mathbf{y}) \| p_\theta(\mathbf{z} \mid \mathbf{y})\right)=\sum_{j=1}^m D_{K L}\left(q_\phi\left(z_j \mid \mathbf{x}, \mathbf{y}\right) \| p_\theta\left(z_j \mid \mathbf{y}\right)\right)	\tag{16}
$$

This is a consequence of conditional independence of the components of $\mathbf{z}$.







### (c) implement CVAE

[12 points] Fill in code for CVAE network as a nn.Module class called CVAE in the starter code cvae.py and the notebook cvae.ipynb:

- Implement the recognition_model function $q_\phi(\mathbf{z} \mid \mathbf{x}, \mathbf{y})$.
- Implement the generative_model function $p_\theta(\mathbf{x} \mid \mathbf{z}, \mathbf{y})$.
- Implement the forward function by inferring the Gaussian parameters using the recognition model, sampling a latent variable using the reparametrization trick and generating the data using the generative model.
- Implement the variational lowerbound loss_function $\mathcal{L}(\theta, \phi ; \mathbf{x}, \mathbf{y})$.
- Train the CVAE and visualize the generated image for each class (i.e., 10 images per class).
- Repeat the image generation 10 times with different random noise. In the write-up, attach and submit $10 \times 10$ array of images showing all the generated images, where the images in the same row are generated from the same random noise, and images in the same column are generated from the the same class label.
- The hyperparameters and training setups provided in the code should work well for learning a CVAE on the MNIST dataset, but please feel free to make any changes as needed and you think appropriate to make CVAE work. Please discuss (if any) there are some notable changes you have made.

If trained successfully, you should be able to sample images $\mathbf{x}$ that look like MNIST digits reflecting the given label $\mathbf{y}$, and the noise vector $\mathbf{z}$.
