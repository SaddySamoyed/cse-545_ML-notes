# lec 11: overview of NN

DL 本质上就是**将 input Space 到 Feature Space 的映射从 single layer (比如 logistic regression) 扩展为 multilayer**，并通过这些层次化的表示来学习数据的更高阶特征.

![image-20250305002815432](04-neural_network.assets/image-20250305002815432.png)

比方说 DL 一个分类问题:

1. 低层神经元：学习边缘、纹理等低级特征。
2. 中层神经元：学习更复杂的模式，如耳朵、眼睛等部件。
3. 高层神经元：学习全局信息，如“这是一只猫”。
4. 最终输出分类结果（猫/狗）。



以下为各种 Machine Learning methods 以 supervised 与否、deep 与否这两个标准的分类

![image-20250305003035572](04-neural_network.assets/image-20250305003035572.png)





### types of neurons

![image-20250305003246906](04-neural_network.assets/image-20250305003246906.png)



![image-20250305003259165](04-neural_network.assets/image-20250305003259165.png)

![image-20250305003512806](04-neural_network.assets/image-20250305003512806.png)





![image-20250305003525036](04-neural_network.assets/image-20250305003525036.png)













### train NN: forward 和 backward propagation

Algorithm of NN training: 

![image-20250305003555313](04-neural_network.assets/image-20250305003555313.png)



在这一过程中, 其中一个 layer 到另一个 layer 是这样啊

![image-20250305004949858](04-neural_network.assets/image-20250305004949858.png)



对于某一层的 parameters $\theta$ 的导数的计算, 可以由 chain rule 得到. 

![image-20250305005738165](04-neural_network.assets/image-20250305005738165.png)

其 vectorization form: 

![image-20250305010120659](04-neural_network.assets/image-20250305010120659.png)

#### my remark: ml 里数学 notation 的不严谨之处, 以及如何指认.

我们首先复习正常的多元实分析中学到的 notation:

![image-20250305010319973](04-neural_network.assets/image-20250305010319973.png)

而 ml 中, 经常会有滥用 notation 的情况。我这里并不是指 notation 的形式和数学里不一样，而是 ml 里经常会有 notation 内部的 inconsistency. 比如下面:

![Screenshot 2025-03-05 at 01.15.28](04-neural_network.assets/Screenshot 2025-03-05 at 01.15.28.png)

我们如何 consistently 翻译这个内容：在 ml 的 notation 里，**对于 R^n to R^m (where m>1)的函数, 不论它写作 nabla 形式还是 partial partial 形式, 都是 Jacobian matrix, 也就是 derivative (即便写成 nabla)；对于 R^n to R 的函数, 不论它写作 nabla 形式还是 partial partial 形式, 都是 gradient, 也就是 derivative 的 transpose (即便写成partial partial)**

![image-20250305011240234](04-neural_network.assets/image-20250305011240234.png)

虽然我们自己仍然严格地使用正确的数学标记(对于 derivative 用 partial partial, 对于 gradient 用 nabla), 但是碰到不好的标记就用上面这个翻译来理解。







#### ex: NN with 1 hidden layer

![image-20250305122536288](04-neural_network.assets/image-20250305122536288.png)

Scalar-valued, 很容易算

![image-20250305122547854](04-neural_network.assets/image-20250305122547854.png)

vectorized: 有点难. 

![image-20250305122619418](04-neural_network.assets/image-20250305122619418.png)

至于 matrix input, vector valued 的函数, 其涉及 tensor 的 differentiation, 目前还没写.

![image-20250305122641797](04-neural_network.assets/image-20250305122641797.png)

Neural Network 对于数据量很大的数据集 work well, 因为 multi layers 可以学习到复杂的数据范式；但是当数据量比较小的时候，它不 work well，反而是传统的 ml 更加有用。因为数据量小的时候它很容易过拟合。（参数过多）







### sgd 一系列的 optimization methods 

在深度学习中，优化算法用于调整模型参数以最小化损失函数。常见的优化算法包括 **SGD（随机梯度下降）**、**Momentum（动量）** 和 **Adagrad（自适应梯度算法）**，它们在更新参数时具有不同的策略。以下是它们的介绍：

#### **1. SGD（随机梯度下降）**

**SGD 是最基本的优化方法。**

- **更新公式**：
  $$
  \theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
  $$
  其中：
  - $ \theta_t $ 是参数
  - $ \eta $ 是学习率
  - $ \nabla L(\theta_t) $ 是损失函数关于参数的梯度

- **特点**：
  - 计算简单，每次仅使用**一个或一小批**样本来计算梯度，减少计算成本。
  - 由于随机梯度的方差较大，**更新方向不稳定**，容易震荡，尤其是在接近最优点时。
  - 可能会陷入**局部极小值**，特别是在复杂的损失面上。



#### **2. Momentum 动量优化**

**Momentum 通过引入“动量”来加速收敛并减少震荡。**

- **更新公式**：
  $$
  v_t = \beta v_{t-1} - \eta \nabla L(\theta_t)
  $$
  $$
  \theta_{t+1} = \theta_t + v_t
  $$
  其中：
  - $ v_t $ 是动量项，累积了过去的梯度信息
  - $ \beta $ 是动量系数，通常取 0.9 左右

- **特点**：
  - 通过**累积过去的梯度**，在梯度方向一致时加速收敛。
  - 在震荡较大的方向（例如损失面狭长的区域），减少梯度波动，提高稳定性。
  - 类似物理学中的“惯性效应”，在陡峭的方向上快速收敛，在平坦方向上平稳前进。

https://distill.pub/2017/momentum/



#### **3. Adagrad（自适应梯度算法）**

**Adagrad 通过调整每个参数的学习率，使得常见特征的学习率降低，而稀疏特征的学习率较高。**

- **更新公式**：
  $$
  G_t = G_{t-1} + \nabla L(\theta_t) \odot \nabla L(\theta_t)
  $$
  $$
  \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla L(\theta_t)
  $$
  其中：
  - $ G_t $ 是累积平方梯度
  - $ \epsilon $ 是一个小常数（如 $ 10^{-8} $），用于数值稳定性

- **特点**：
  - 适用于**稀疏数据**（如 NLP 任务），能够自适应调整学习率，使得更新更稳定。
  - **问题**：梯度平方累积不断增大，导致学习率逐渐趋近于 0，可能会过早停止学习。



在实际应用中，**Momentum 比 SGD 更受欢迎**，因为它能加速收敛并减少震荡。而 **Adagrad 适合稀疏数据，但学习率下降过快的问题常用 RMSprop（对 Adagrad 进行了改进）来解决**。

你如果对 RMSprop 或 Adam（结合了 Momentum 和 Adagrad）也感兴趣，可以继续探讨！



RMSprop 和 Adam 是优化算法的改进版本，它们在深度学习中广泛使用，因为它们能有效地加速训练并提高收敛稳定性。以下是详细介绍：

#### 4. RMSprop（Root Mean Square Propagation）
RMSprop 是对 Adagrad 的改进，解决了其学习率随时间下降过快的问题。

- Adagrad 问题：
  - Adagrad 在训练过程中会不断累积历史梯度平方 $ G_t $，导致学习率 $ \eta / \sqrt{G_t} $ 逐渐变小，从而使得训练提前停滞。
  - 解决方案：不累积所有过去的梯度，而是使用指数加权移动平均（EMA） 让过去的梯度衰减。

- 更新公式：
  $$
  S_t = \beta S_{t-1} + (1 - \beta) \nabla L(\theta_t)^2
  $$
  $$
  \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{S_t + \epsilon}} \nabla L(\theta_t)
  $$
  其中：
  - $ S_t $ 是梯度平方的指数加权移动平均（EMA）。
  - $ \beta $ 是衰减系数（通常取 0.9）。
  - $ \epsilon $ 是一个小数（如 $ 10^{-8} $），避免分母为零。

- 特点：
  - 与 Adagrad 不同，**RMSprop 只保留最近一段时间的梯度信息，不会让学习率无限衰减**。
  - 适用于非平稳目标函数（如深度学习）。
  - 适用于RNN（循环神经网络），因为它能处理不同尺度的梯度变化。



#### 5. Adam（Adaptive Moment Estimation）

Adam 结合了 Momentum 和 RMSprop 的思想，是目前最流行的优化算法之一。

- Adam 结合了两个部分：
  1. Momentum 机制（梯度的指数加权移动平均），用于加速收敛：
     $$
     m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(\theta_t)
     $$
     其中 $ m_t $ 是梯度的一阶矩估计，相当于梯度的动量。
  
  2. RMSprop 机制（梯度平方的指数加权移动平均），用于自适应学习率：
     $$
     v_t = \beta_2 v_{t-1} + (1 - \beta_2) \nabla L(\theta_t)^2
     $$
     其中 $ v_t $ 是梯度的二阶矩估计，相当于梯度平方的滑动平均。

  - 偏差校正（解决初始动量偏差）：
    $$
    \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
    $$
  - 最终参数更新：
    $$
    \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
    $$

- 超参数选择：
  - 默认推荐值：
    - $ \beta_1 = 0.9 $（动量项）
    - $ \beta_2 = 0.999 $（梯度平方项）
    - $ \eta = 0.001 $（学习率）

- 特点：
  - 结合了 Momentum 和 RMSprop 的优点，既能加速收敛，又能自适应调整学习率。
  - 对超参数不敏感，默认参数通常表现很好。
  - 广泛用于各种深度学习任务，包括 CNN、RNN 和 Transformer。

---

#### 对比总结以及选择

https://github.com/j-w-yun/optimizer-visualization

| 优化算法     | 优势                                                   | 劣势                                  |
| :----------- | ------------------------------------------------------ | ------------------------------------- |
| **SGD**      | 简单高效，适用于大规模数据                             | 更新方向不稳定，可能收敛慢            |
| **Momentum** | 减少梯度震荡，加速收敛                                 | 需要调节额外的动量参数 $ \beta $      |
| **Adagrad**  | 适用于稀疏数据，能自适应调整学习率                     | 学习率会随时间下降，可能导致训练停滞  |
| **RMSprop**  | 解决 Adagrad 过快衰减学习率的问题，适用于 RNN          | 仍然需要手动调节学习率                |
| **Adam**     | 结合 Momentum 和 RMSprop，收敛快且稳定，对超参数不敏感 | 可能在某些任务上表现不如 SGD+Momentum |

- **SGD + Momentum：适用于大规模深度学习任务，如 CNN 训练**，能提供更好的泛化能力（避免 Adam 过拟合）。
- **RMSprop：适用于RNN 训练，比如 LSTM 处理时间序列数据**。
- **Adam：适用于大多数任务，默认推荐的优化器，特别是 NLP 和 Transformer 任务**。



### batch normalization

**Batch Normalization（BN）** 是一种用于**加速神经网络训练并提高稳定性**的技术，特别是在深度网络中，它能减少**内部协变量偏移（Internal Covariate Shift）**，使训练更快、更稳定。

---

为什么需要 Batch Normalization？
在深度神经网络中，不同层的输入分布可能会随着训练的进行不断变化，这被称为**内部协变量偏移**。这种变化会导致：

- **梯度消失或梯度爆炸**（特别是在深层网络中）。
- 需要**更小的学习率**，否则容易震荡或不收敛。
- 依赖**精细的权重初始化**，否则优化困难。

BN 通过在**每一层的输入上做归一化**，让数据分布保持稳定，从而缓解这些问题。

---

假设某一层的输入是 $ x $，Batch Normalization 计算如下：

1. **计算 mini-batch 均值和方差：**
   $$
   \mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i
   $$
   $$
   \sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2
   $$
   其中，$ m $ 是 mini-batch 的大小，$ \mu_B $ 和 $ \sigma_B^2 $ 分别是该 batch 的均值和方差。

2. **归一化（标准化）：**
   $$
   \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
   $$
   其中 $ \epsilon $ 是一个很小的常数（如 $ 10^{-5} $），防止分母为 0，for numerical stability

3. **缩放和平移（可训练参数）：**
   $$
   y_i = \gamma \hat{x}_i + \beta
   $$
   - $ \gamma $（缩放参数）和 $ \beta $（平移参数）是**可学习的参数**，用于恢复模型的表达能力。
   - 经过 BN 后，数据分布被归一化，但仍然保留了模型的非线性变换能力。

---

BN 的优点
✅ **提高训练速度**：减少梯度消失/爆炸问题，可以使用**更大的学习率**，加速收敛。  
✅ **减小参数对初始化的依赖**：不需要精细的权重初始化，训练更加稳定。  
✅ **缓解过拟合**：由于 BN 在 mini-batch 级别上引入了一定的噪声，类似于 Dropout，具有一定的正则化效果。  

---



BN 在训练和推理阶段的区别:

在**训练阶段**：

- BN 使用 mini-batch 的均值和方差进行归一化。
- 由于不同 batch 的均值和方差可能有所不同，会引入一定的噪声。

在**推理阶段**：
- 不能依赖 mini-batch 进行归一化，而是使用整个训练集的**全局均值和方差**：
  $$
  \mu_{\text{global}} = \frac{1}{N} \sum_{\text{all batches}} \mu_B
  $$
  $$
  \sigma_{\text{global}}^2 = \frac{1}{N} \sum_{\text{all batches}} \sigma_B^2
  $$
- 这些全局统计量是在训练过程中累计计算的，推理时不会再使用 batch 内均值。

---



Batch Normalization vs 其他归一化方法：

| 方法                            | 归一化范围                 | 计算方式                    | 适用场景                       |
| ------------------------------- | -------------------------- | --------------------------- | ------------------------------ |
| **Batch Normalization (BN)**    | 每个 mini-batch            | 计算 batch 内均值/方差      | CNN、DNN                       |
| **Layer Normalization (LN)**    | 每个样本的所有特征维度     | 计算单个样本的均值/方差     | NLP（如 Transformer）          |
| **Instance Normalization (IN)** | 每个样本的单个通道         | 计算单个通道内的均值/方差   | 风格迁移、图像任务             |
| **Group Normalization (GN)**    | 每个样本的多个通道（分组） | 计算某一组通道内的均值/方差 | CNN（特别是 batch 小的情况下） |

---

BN 的局限性：
❌ **对 batch size 敏感**：如果 batch size 很小，BN 计算的均值和方差不稳定，可能导致模型性能下降（可以用 GN 代替）。  
❌ **不适用于 RNN**：在序列任务（如 RNN）中，BN 的 batch 计算方式不适用，通常用 **Layer Normalization (LN)** 替代。  
❌ **计算开销增加**：BN 需要计算均值和方差，并存储额外的统计信息，会增加计算量，特别是在推理阶段。





# lec 12: CNN

### image classification if fully connected

![image-20250306053559243](04-neural_network.assets/image-20250306053559243.png)







### convolution layer

Convolution layer 的 idea: 利用 pixels 的 2D topology, 即: 图形的 translate invariance 等.

具体:

1. 利用 Local connectivity,
2. parameter sharing
3. pooling, subsampling hidden units.



#### convolution layer: activation map

![image-20250306054552189](04-neural_network.assets/image-20250306054552189.png)

![image-20250306072411570](04-neural_network.assets/image-20250306072411570.png)

多个 filters 可创建多个 activation maps

![image-20250306072606467](04-neural_network.assets/image-20250306072606467.png)





#### express the feature map as convolution

![image-20250306073659842](04-neural_network.assets/image-20250306073659842.png)

ex: ![image-20250306074514773](04-neural_network.assets/image-20250306074514773.png)



#### hyperparameter: strides

convolution layer 一共有 4 个 hyperparameters: 

1. number of filters
2. filter spatial dimension
3. stride
4. pad 

改变 strides 即改变 convolution 采样的间距. stride 越大, 表示采样越稀疏, 降低对了特征的采集的精细度.

![image-20250306075148349](04-neural_network.assets/image-20250306075148349.png)

计算 activation map 的像的维度的公式:

![image-20250306155330378](04-neural_network.assets/image-20250306155330378.png)



#### hyperparameter: padding

**`padding`（填充）**即在 input image 的边缘添加额外的像素，以控制输出的空间维度并提升模型的性能。它的作用主要包括以下几个方面：

1. **控制输出尺寸**

   我们已经知道，kernel 会减少输入特征图的尺寸 (如果 stride =1, 减少的是 kernel 的尺寸; 如果 stride 更大, 减少地更多 )。通过**填充（padding）**，可以按照需求调整输出尺寸，比如使输出特征图的大小保持与输入相同（如`same padding`）

2. **保留边界信息**

   在没有填充的情况下，**靠近边缘的像素会比中心的像素被卷积核覆盖的次数少**，导致边缘信息容易丢失。`padding` 允许卷积核在边界区域也能进行相同次数的计算，提升模型对边缘特征的学习能力。

3. **保证卷积核中心对齐**

   当使用**奇数大小的卷积核**（如 $3 \times 3$, $5 \times 5$）边界像素的计算会相对不对称. `padding` 可以使得所有位置都能完整地进行卷积运算，保证特征提取的一致性。



**计算 output 大小:**

![image-20250306160932346](04-neural_network.assets/image-20250306160932346.png)

(for one kernel.)





常见的 Padding 方式是 **Same Padding 或称 zero padding**（`padding` 使得输出尺寸等于输入尺寸）

- **步长 $ \text{stride}=1 $** 时：
  $$
  \text{padding} = \frac{k - 1}{2}, \quad \text{(当 k 为奇数时)}
  $$
  例如：

  - $3 \times 3$ 卷积核 → `padding=1`
  - $5 \times 5$ 卷积核 → `padding=2`

![Screenshot 2025-03-06 at 16.02.30](04-neural_network.assets/Screenshot 2025-03-06 at 16.02.30.png)



可以人为设定不同方向上的填充，例如 TensorFlow/PyTorch 允许 `(top, bottom, left, right)` 方式填充。在实际 CNN 设计中，`same padding` 更常用于深度网络，以保证特征图尺寸稳定。



### pooling layer

`Pooling` 池化是 CNN 中除了 convolution layer 外的另一个常用 layer. 它的 parameter 是 window size 和 stride. 

convolution layer 通过滑动窗口和 image 对应位置作 inner product 来提取特征图，而 pooling 则是一种更加直接的 **subsampling (降采样)**，主要目的是降低特征图的尺寸，减少计算量，同时保留重要特征。同样**利用了模型的平移不变性，减少参数数量，从而提高泛化能力。**

![image-20250306163050853](04-neural_network.assets/image-20250306163050853.png)



**常见的 pooling 操作**：

#### Max Pooling

- 计算窗口区域内的最大值。

- 主要用于保留局部区域的最强特征（高激活值）。

- 适用于提取边缘和纹理等重要特征，常用于 CNN 任务，如图像分类、目标检测等。

  ![image-20250306163239902](04-neural_network.assets/image-20250306163239902.png)

#### Average Pooling

- 计算窗口区域内所有值的**平均值**。

- 适用于平滑特征图，减少过拟合，但可能会导致信息损失较大。

- 在某些任务（如图像分割、强化学习）中仍然有用。

  ![image-20250306163255277](04-neural_network.assets/image-20250306163255277.png)

  



#### Global Average Pooling, GAP

- 直接对整个特征图计算**全局平均值**，得到一个数值（每个通道一个）。
- 主要用于替代全连接层，减少参数量。
- 在**ResNet、MobileNet**等轻量级 CNN 结构中经常使用。



#### **池化 vs 卷积**

Note: pooling 是不可训练的，完全是固定的。它没有参数，全部都是超参数

| **对比项**   | **池化 (Pooling)**   | **卷积 (Convolution)** |
| ------------ | -------------------- | ---------------------- |
| **作用**     | 降采样，减少计算量   | 提取特征               |
| **参数**     | 无参数               | 具有可学习的权重       |
| **影响**     | 提高模型的平移不变性 | 主要用于提取特征       |
| **可训练性** | 不可训练             | 可训练                 |



虽然池化很常见，但一些现代 CNN 结构（如 ResNet、DenseNet）倾向于用 **stride 卷积** 代替 `pooling` 来进行降维，因为：

- `stride` 卷积可以学习降维方式，而池化是固定操作。
- `stride` 卷积允许模型在降维的同时提取更多特征，提高表达能力。

但在轻量级模型（如 MobileNet）或某些任务（如语音处理、医学影像分析）中，**池化仍然是有效的降维手段**。













### CNN 架构

#### classical 架构

![Screenshot 2025-03-06 at 16.35.24](04-neural_network.assets/Screenshot 2025-03-06 at 16.35.24.png)

ex: vehicle 分类

![Screenshot 2025-03-06 at 16.34.38](04-neural_network.assets/Screenshot 2025-03-06 at 16.34.38.png)





#### Lenet

![Screenshot 2025-03-06 at 16.35.52](04-neural_network.assets/Screenshot 2025-03-06 at 16.35.52.png)



#### Alexnet

![Screenshot 2025-03-06 at 16.38.51](04-neural_network.assets/Screenshot 2025-03-06 at 16.38.51.png)



#### VGG-16

![Screenshot 2025-03-06 at 16.39.36](04-neural_network.assets/Screenshot 2025-03-06 at 16.39.36.png)





#### ResNet

![Screenshot 2025-03-06 at 16.42.07](04-neural_network.assets/Screenshot 2025-03-06 at 16.42.07.png)

![Screenshot 2025-03-06 at 16.42.24](04-neural_network.assets/Screenshot 2025-03-06 at 16.42.24.png)

![Screenshot 2025-03-06 at 16.42.53](04-neural_network.assets/Screenshot 2025-03-06 at 16.42.53.png)



error 排名：

![Screenshot 2025-03-06 at 16.43.51](04-neural_network.assets/Screenshot 2025-03-06 at 16.43.51.png)





### CNN 应用

#### object classification  

![Screenshot 2025-03-06 at 16.50.43](04-neural_network.assets/Screenshot 2025-03-06 at 16.50.43.png)

这是最简单的应用



#### object detection: proposing bounding boxes



![Screenshot 2025-03-06 at 16.53.44](04-neural_network.assets/Screenshot 2025-03-06 at 16.53.44.png)

RCNN: 

![Screenshot 2025-03-06 at 16.54.36](04-neural_network.assets/Screenshot 2025-03-06 at 16.54.36.png)

Fast RCNN:

![Screenshot 2025-03-06 at 16.55.15](04-neural_network.assets/Screenshot 2025-03-06 at 16.55.15.png)



#### semantic segmentation

![Screenshot 2025-03-06 at 16.56.20](04-neural_network.assets/Screenshot 2025-03-06 at 16.56.20.png)

![Screenshot 2025-03-06 at 16.56.41](04-neural_network.assets/Screenshot 2025-03-06 at 16.56.41.png)





### Transfer learning and fine-tuning

**Transfer Learning（迁移学习）** 和 **Fine-Tuning（微调）** 都是基于 **预训练模型（Pre-trained models）** 的方法，用于在新的任务上高效地训练深度学习模型，特别是在计算资源有限或数据量不足的情况下。

Transfer learning 的核心思想是使用**已经在大规模数据集（如 ImageNet）上训练好的模型**，然后将其应用到**新的但相似的任务**上，而不需要从零开始训练整个网络。

- **不调整预训练模型的大部分参数**，而是**利用其提取的特征**。
- 一般只**移除最后几层**，替换为新的分类层，并在新数据集上训练新的分类器。
- 适用于**数据量较少**的情况，能够**快速收敛**。

**主要步骤**:

1. **使用预训练模型的权重**（例如 ResNet、VGG、EfficientNet）。
2. **移除最后的分类层**（通常是全连接层）。
3. **使用网络的前面部分作为特征提取器**（冻结这些层的权重）。
4. **添加新的分类头（通常是一个全连接层）** 并在新的任务数据上训练。

**应用场景**

- 计算机视觉：图像分类、目标检测（如从 ImageNet 迁移到医学影像分析）。
- 自然语言处理（NLP）：使用预训练的 BERT 或 GPT 进行情感分析等任务。





**Fine-Tuning（微调）** 是迁移学习的一种**更深入的**方法，不仅仅是利用预训练模型的特征提取能力，还**对部分或全部预训练模型的参数进行调整**，以适应新任务。

1. **去掉预训练模型的最后几层**。
2. **添加新的分类头**，先训练这个部分（类似迁移学习）。
3. **解冻部分深层（或整个网络）的参数**，使用较小的学习率进行训练（防止破坏预训练的权重）。
4. **训练整个网络**，进一步优化新任务上的性能。

适用于任务和预训练数据集的相似度较高时（如猫狗分类 → 狼狗分类）。



| 特性             | 迁移学习 (Transfer Learning)                            | 微调 (Fine-Tuning)                              |
| ---------------- | ------------------------------------------------------- | ----------------------------------------------- |
| **冻结层数**     | 大部分层被冻结，仅训练新分类层                          | 可能解冻部分或全部层进行训练                    |
| **计算资源需求** | 低                                                      | 高                                              |
| **数据需求**     | 少                                                      | 需要更多数据                                    |
| **适用场景**     | 任务不同但特征相似（如 ImageNet 迁移到 X-Ray 影像分类） | 任务相似，需更精确调整（如猫狗分类 → 狼狗分类） |
| **训练时间**     | 较短                                                    | 较长                                            |





