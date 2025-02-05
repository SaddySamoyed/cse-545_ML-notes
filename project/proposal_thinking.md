# Data-centric methods to improve CLIP-based multimodal representation learning

## 前提资料

### 0.什么是 CLIP: CLIP (Contrastive Language-Image Pre-Training) 

CLIP（对比语言-图像预训练），一种在不同图像-文本对上进行训练的神经网络，可以使用自然语言指令预测给定图像的相关文本，展示与 GPT 模型类似的零拍摄能力。它通过在共享的嵌入空间中对图像和文本进行对齐，学习如何共同理解图像和文本。CLIP 使用大量图像-文本对数据集，**通过 InfoNCE 对比目标训练视觉模型（如 ResNet 或 Vision Transformer）和文本模型（如 Transformer），以预测哪张图像与给定的文本描述相匹配**，反之亦然。这种对比学习方法利用其连接视觉和文本概念的能力，使 CLIP 能够很好地通用于各种任务，如图像分类或  zero-shot learning，而无需针对特定任务进行 fine-tuning。

CLIP 在不使用标注示例的情况下，与 ResNet50 在 ImageNet 上的零拍摄设置性能不相上下，这凸显了它在没有特定任务训练的情况下跨任务泛化的能力。最近的一些研究尤其表明，**CLIP 性能与训练目标难度之间存在正相关。**

### **1. 什么是 MultiModel**

**MultiModel（多模型）** 指的是在机器学习或深度学习中使用多个不同的模型来解决一个任务。这些模型可以是：

- **不同架构的模型**（如 CNN + RNN 组合）
- **相同架构但不同参数的模型**（如多个 CNN 处理同一任务）
- **用于不同模态的数据**（如图像、文本、音频等）

MultiModel 主要用于：

- **模型集成（Ensemble Learning）**：如集成多个模型进行投票或加权平均，提高预测准确率。
- **混合专家（Mixture of Experts, MoE）**：不同的子模型处理不同的任务或输入。
- **多模态学习（Multimodal Learning）**：不同的模型处理不同模态的数据，如 GPT-4 既能处理文本，也能处理图像输入。

**示例：**

- Google 的 **MultiModel**：一个统一的深度学习架构，可以同时处理图像、语音和文本。
- **AlphaFold**：结合多个深度学习模型预测蛋白质结构。

------

### **2. 什么是 Representation Learning（表征学习）**

**Representation Learning（表示学习）** 是指自动学习数据的有用表示（特征），使得模型可以更好地进行分类、回归或生成任务。表示学习的目标是从数据中提取高层次、可泛化的信息，而不是直接使用原始数据。

#### **表示学习的主要方法**

1. **自监督学习（Self-supervised Learning）**
   - 通过数据本身构造训练目标（如 BERT 的掩码语言模型）。
   - 示例：GPT、SimCLR（用于计算机视觉）。
2. **对比学习（Contrastive Learning）**
   - 通过最大化正样本的相似性、最小化负样本的相似性来学习表示。
   - 示例：SimCLR、MoCo、CLIP。
3. **自动编码器（Autoencoder, AE）**
   - 通过压缩和重构数据来学习低维表示。
   - 示例：Variational Autoencoder（VAE）。
4. **深度嵌入（Deep Embedding）**
   - 例如 Word2Vec、BERT 将文本转换为向量表示。
   - 例如 ResNet 提取图像特征向量。

#### **表示学习的应用**

- **自然语言处理（NLP）**：如 BERT、GPT 进行文本表示学习。
- **计算机视觉（CV）**：如 CNN 提取图像特征。
- **多模态任务**：如 CLIP 学习文本-图像对齐的表示。



| 术语                                    | 解释                                       | 主要应用                                  |
| --------------------------------------- | ------------------------------------------ | ----------------------------------------- |
| **MultiModel（多模型）**                | 组合多个不同模型来提高性能或解决不同任务   | 集成学习、专家混合、多模态学习            |
| **Representation Learning（表示学习）** | 让模型自动学习数据的有效表示，提高泛化能力 | NLP（BERT）、CV（ResNet）、多模态（CLIP） |

MultiModel 主要关注模型的组合和优化，而 Representation Learning 关注如何让模型学习有意义的特征。











## general purpose 以及现有的 data-centric methods

Yuksekgonul 等人（2022 年）的研究表明，模型在关系、归属和顺序理解方面存在困难，潜在的解决方案可能是对训练程序进行修改，即 “成分感知硬负面挖掘”。另外，也有团队采用 “构建包含相似性集群的迷你批次以增加负面示例的难度 ”的方法来解决问题。(Liu等人，n.d.）。除此之外，通过语言重写增加数据多样性也显示出了提高性能的潜力（Fan 等人，2023 年）。



研究表明，可以通过增加训练目标难度来提高 CLIP 的性能。当负图像对与正图像对更难区分时，CLIP 模型就会被迫学习表现图像中更多的微妙特征。为此，人们开发了几种以数据为中心的方法，包括将更多相似的图像-文本对放入一个批次（[NegCLIPLinks to an external site.](https://arxiv.org/pdf/2210.01936)、[SimCLIPLinks to an external site.](https://openreview.net/pdf?id=NmNmlAEBAl)）或通过数据增强引入硬否定（[NegCLIPLinks to an external site.](https://arxiv.org/pdf/2210.01936)、[CE-CLIPLinks to an external site.](https://arxiv.org/pdf/2306.08832)、[SLVCLinks to an external site.](https://arxiv.org/pdf/2211.11733)）。这些方法通常在自然图像数据集（如 Imagenet 或 MSCOCO）以及专为评估模型对构成性的理解而设计的较难数据集（[AROLinks to an external site.](https://huggingface.co/datasets/gowitheflow/ARO-Visual-Attribution)、[VALSELinks to an external site.](https://paperswithcode.com/dataset/valse)、[VL-CheckListLinks to an external site.](https://github.com/om-ai-lab/VL-CheckList)、[SugarCrepeLinks to an external site.](https://github.com/RAIVNLab/sugar-crepe)）上进行基准测试。此外，众所周知，简单的数据扩充可以提高数据多样性（即使不增加难度），从而提高 perfomance（例如 [LLM 文本重写链接到外部网站](https://arxiv.org/abs/2305.20088)）。

In this project, students are encouraged to explore or design new data-centric methods or augmentations to improve CLIP performance on **small training datasets.** 







以下是老师提供的 data centric methods

### 方法1.putting more similar image-text pairs into one batch

#### ex1：Negclip

尽管大型视觉和语言模型（VLMs）在许多下游应用中得到了使用，how well they encode the compositional relationships between objects and attributes 仍待探究。在此，我们创建了**Attribution, Relation, and Order (ARO) benchmark**，以系统地评估 VLM 理解不同类型的关系、属性和顺序信息的能力。

ARO 由视觉基因组归属（Visual Genome Attribution）、视觉基因组关系（Visual Genome Relation）和 COCO-Order & Flickr30k-Order 组成，前者用于测试对对象属性的理解，后者用于测试对关系的理解，而 COCO-Order & Flickr30k-Order 则用于测试 VLM 对顺序的敏感性。

ARO 的规模比以前的组合性基准大很多，有 50,000 多个测试案例。我们介绍了最先进的 VLM 在哪些情况下会表现得像词袋，即它们对关系的理解能力较差，在将对象与其属性联系起来时可能会出错，并且严重缺乏对顺序的敏感性。VLM 主要是在图像和标题具有丰富组成结构的大型数据集上进行训练和评估。然而，在这些数据集上进行的训练并不足以解决缺乏构图理解的问题，在这些数据集上进行的评估也未能发现这一缺陷。为了了解为什么会出现这些局限性，而且这些局限性没有在标准测试中体现出来，我们对训练和评估程序进行了深入研究。我们证明，在现有的数据集上，不使用构图和顺序信息也能很好地进行图像-文本检索。这进一步激发了使用 ARO 作为 VLM 基准的价值。鉴于 contrastive pretraining 可以优化具有类似捷径的大型数据集上的检索，我们假设这可以解释为什么模型不需要学习表示成分信息。这一发现提出了一个自然的解决方案：c**omposition-aware hard negative mining**。我们的研究表明，对对比学习进行一个 simple-to-implement modification，就能显著提高需要理解顺序和构成性的任务的性能。



#### ex2：SimClip:  similarity-based CLIP fine-tuning framework

similarity clusters(就是 honglak 本人做的): 随着 CLIP 训练在学习可迁移视觉表征（ transferable visual representations）方面的成功，**在较小的数据集上微调 CLIP 模型以获得更好的下游性能成为一个重要的研究领域。**改进 CLIP 模型的一种方法是增加负面示例的难度。虽然**大多数研究都集中在 manually crafting hard negative captions** 上，但这种策略需要额外的工程，无法推广到不同的领域，还会导致额外的过拟合。在此，我们进行了一项实证研究，系统地探索了一种替代方法：**构建包含 similarity clusters 的 minibatches，以增加 the difficulty of negative example**。

我们提出了一个 generalized framework, 称为 SimCLIP，用于 similarity-based CLIP fine-tuning。通过强制要求每个 minibatch 包含相似示例集群，SimCLIP fine-tuning（微调）与标准 CLIP  fine-tuning  相比可以提高模型性能。我们广泛研究了哪些 SimCLIP 配置和因素对下游性能贡献最大。我们还分析了 SimCLIP 在罕见特殊集合、compositionality of attributes 以及泛化性能上的表现方面，对于不同数据集大小的不同。

代码: https://github.com/sx-liu/SimCLIP/



### 方法2: introducing hard negatives via data augmentation

（negclip 也做了这一点，在上）

#### ex3: **Ce-clip**

Contrasting intra-modal and ranking cross-modal hard negatives to enhance visio-linguistic compositional understanding

概要: Vision-Language Models (VLMs) 如 CLIP 有强大的 mage-text comprehension abilities, 促进了一些 downstream tasks 的进展, 比如 zero-shot image classification, image-text retrieval, and text-to-image generation. 然而，现有 VLM 的构图推理能力仍不尽如人意。这种**局限性的根源在于预训练数据集中的 images 和 captions 之间的 alignment  不够**。此外，当前的对比学习目标**未能关注 fine-grained grounding components like relations, actions, and attributes**，从而导致了 "bag-of-words" representations。

我们介绍了一种简单有效的方法来**改进 compositional reasoning in VLMs**：通过 **refining and expanding the standard image-text contrastive learning framework**，更好地利用了现有数据集。我们的方法不需要特定的注释，也不会产生额外的参数。当与 CLIP 整合时，我们的技术在五个视觉语言合成基准中的表现明显优于最先进的基线技术



#### ex4: teaching structured VLM

我们引入了 Structured Vision & Language Concepts (SVLC) “结构化视觉与语言概念”（SVLC）这一 collective notion，其中包括文本中存在、图像中可见的 object attributes, relations, and state。最近的研究表明，**即使是最好的视觉语言模型也很难处理 SVLC**。解决这一问题的一个可行方法是 collecting dedicated datasets for teaching each SVLC type，但这可能既昂贵又耗时。

相反，我们提出了一种更优雅的数据驱动方法来增强 VL 模型对 SVLC 的理解，这种方法能更有效地利用现有的 VL 预训练数据集，而且不需要任何额外的数据。图像结构的自动理解在很大程度上仍未得到解决，不过语言结构的建模和理解则要好得多，因此 allowing for its effective utilization in teaching VL models.。

本文提出了多种基于语言结构理解的技术，可用于处理现成配对 VL 数据集的文本部分。无论是从头开始训练还是微调预先训练的模型，使用更新数据训练的 VL 模型在 SVLC 理解能力方面都有显著提高，最高可达 15%，而在零拍摄能力方面只有轻微下降。

代码：https://github.com/SivanDoveh/TSVLC







### 方法3. data augmentation that improves data diversity

#### ex5: LaClip

代码：  https://github.com/LijieFan/LaCLIP.（nips 2023)

CLIP 模型采用 **contrastive loss** 进行训练，通常依靠 data augmentations 来防止过度拟合和走捷径。然而，在 CLIP 训练范式中，**数据增强只应用于图像输入，而语言输入在整个训练过程中保持不变**，这就**限制了不同文本与同一图像的接触**。在本文中，我们介绍了**语言增强 CLIP（LaCLIP）**，这是一种通过语言改写来增强 CLIP 训练的简单而高效的方法。利用大型语言模型的上下文学习能力，我们重写了与每幅图像相关的文本描述。这些重写的文本在句子结构和词汇方面呈现出多样性，同时保留了原有的关键概念和含义。在训练过程中，**LaCLIP 会随机选择原始文本或改写文本作为每幅图像的文本增强**。在 CC3M、CC12M、RedCaps 和 LAION-400M 数据集上进行的大量实验表明，利用语言改写进行的 CLIP 预训练可显著提高传输性能，且在训练过程中不会产生计算或内存开销。特别是在 ImageNet 零点准确率方面，LaCLIP 在 CC12M 数据集上比 CLIP 高出 8.2%，在 LAION-400M 数据集上高出 2.4%。







## 数据集

general:

- Kaggle competitions: [http://www.kaggle.com/](https://www.google.com/url?q=http://www.kaggle.com/&sa=D&source=editors&ust=1738674855353713&usg=AOvVaw2weNfEXkIl_ySKiInASAer)
- EvalAI Challenges: [https://eval.ai/web/challenges/list](https://www.google.com/url?q=https://eval.ai/web/challenges/list&sa=D&source=editors&ust=1738674855354294&usg=AOvVaw0TLbbPPsGLHpcTtZK26Wu8) 
- Datasets that can be used for Deep Learning models (or non-deep-learning models): [http://deeplearning.net/datasets/](https://www.google.com/url?q=http://deeplearning.net/datasets/&sa=D&source=editors&ust=1738674855354652&usg=AOvVaw3wyBIUuCtZfEiG7spMuHxE) [https://paperswithcode.com/datasets](https://www.google.com/url?q=https://paperswithcode.com/datasets&sa=D&source=editors&ust=1738674855354865&usg=AOvVaw0Di4U1n_NNacPnv1R5FE3h) 
- Fake detection datasets: [Deepfake Detection Challenge Dataset](https://www.google.com/url?q=https://ai.facebook.com/datasets/dfdc/&sa=D&source=editors&ust=1738674855355148&usg=AOvVaw3KjKPdpMX-ZoN3dC5UbW_X)
- NLP Datasets (some of these are getting saturated/outdated): [GLUE](https://www.google.com/url?q=https://paperswithcode.com/dataset/glue&sa=D&source=editors&ust=1738674855355442&usg=AOvVaw26QuyPkz3PoUN8y6gBJNuI), [SQuAD](https://www.google.com/url?q=https://paperswithcode.com/dataset/squad&sa=D&source=editors&ust=1738674855355687&usg=AOvVaw0gkmLsWiWhiV0FuzKVgS7G), [CoQA](https://www.google.com/url?q=https://stanfordnlp.github.io/coqa/&sa=D&source=editors&ust=1738674855355954&usg=AOvVaw3Jlh8x1jX0R3udv5bU6ITN), [SuperGLUE](https://www.google.com/url?q=https://super.gluebenchmark.com/&sa=D&source=editors&ust=1738674855356160&usg=AOvVaw0rZfSSaFng1yLp7oTnyF4r), [MMLU](https://www.google.com/url?q=https://github.com/hendrycks/test&sa=D&source=editors&ust=1738674855356460&usg=AOvVaw2OZu-8erc_ut5NAmkL0hTO), [BigBench](https://www.google.com/url?q=https://github.com/google/BIG-bench&sa=D&source=editors&ust=1738674855356812&usg=AOvVaw2-Vk7iIadoc6Z1_6sm-r-p), [Open LLM Leaderboard](https://www.google.com/url?q=https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard&sa=D&source=editors&ust=1738674855357204&usg=AOvVaw3PJwAFFqnnxvMo-1lfLsgW), etc.
- (Deep) RL environments of toy games: [OpenAI Gym](https://www.google.com/url?q=https://gym.openai.com/&sa=D&source=editors&ust=1738674855357701&usg=AOvVaw1KyEby00Et10I3-P0il1mQ), [DM Control Suite](https://www.google.com/url?q=https://www.deepmind.com/open-source/deepmind-control-suite&sa=D&source=editors&ust=1738674855357980&usg=AOvVaw3_C55jvM58N6EG4FTXCQnA), [DeepMind Lab](https://www.google.com/url?q=https://github.com/deepmind/lab&sa=D&source=editors&ust=1738674855358223&usg=AOvVaw2Fl3CbSTOx5Y0h7WdbZKf_), [MuJoCo](https://www.google.com/url?q=https://mujoco.org/&sa=D&source=editors&ust=1738674855358435&usg=AOvVaw3hYQMuRtSjNW8mBA4p0DVW), [FurnitureBench](https://www.google.com/url?q=https://clvrai.github.io/furniture-bench/&sa=D&source=editors&ust=1738674855358707&usg=AOvVaw3xqDADaSpZO8HEFQKOTFWR)
- Network and graph datasets: [http://snap.stanford.edu/data/](https://www.google.com/url?q=http://snap.stanford.edu/data/&sa=D&source=editors&ust=1738674855359252&usg=AOvVaw2P_iXKRZIKpRvnIX40Rik9)

专门用在自然图像数据集（如 Imagenet 或 MSCOCO）以及专为评估模型对构成性的理解而设计的较难数据集：

[AROLinks to an external site.](https://huggingface.co/datasets/gowitheflow/ARO-Visual-Attribution)、[VALSELinks to an external site.](https://paperswithcode.com/dataset/valse)、[VL-CheckListLinks to an external site.](https://github.com/om-ai-lab/VL-CheckList)、[SugarCrepeLinks to an external site.](https://github.com/RAIVNLab/sugar-crepe)





Knowledge-CLIP 通过引入 Knowledge Graph，提高了 CLIP 在语义对齐和推理任务上的表现，够提取实体之间的结构信息，并通过 KD Loss 蒸馏原始 CLIP 模型的知识，防止新任务训练过程中遗忘 CLIP 的原始能力。Knowledge-CLIP 原本是针对大型数据集的模型，依赖于 multimodel transformer, 但我们认为，Knowledge-CLIP 强大的知识驱动语义对齐能力是可以适应小型数据集，并且其**知识图谱增强（KG-enhanced learning）**，能够：

- **补充小数据集中缺乏的关系信息**，避免模型过拟合于有限的数据分布。
- **提高模型对对象属性、关系的理解能力**，使其更具泛化能力。
- **通过先验知识弥补数据不足**，即使训练数据稀少，模型仍然能学习合理的对象属性关联。

我们的计划是以 Knowledge-CLIP 为基础，我们现阶段的计划是基于 Knowledge-CLIP 进行尝试修改，让它能在小型数据集上达到良好的效果。我们打算保留 Knowledge-CLIP 的知识增强机制，但去掉其复杂的多模态 Transformer 结构，并结合 SimCLIP、AHNM 等更高效的优化策略，引入更轻量化的对比学习优化策略。

###  Pipeline

1. **数据增强**
   - **知识图谱增强（KG-Augmented Data）**：用外部知识补充对象、属性、关系信息。
   - **SimCLIP 相似性簇采样**：提高 minibatch 质量，增强 Hard Negatives。
2. **编码器**
   - **ViT-Small 作为图像编码器**
   - **DistilBERT 作为文本编码器**
   - 轻量化的模型结构，减少计算开销。
3. **对比学习**
   - **AHNM 负例采样**：动态选择 Hard Negatives，优化训练。
   - **Fair Contrastive Loss (FCL)**：优化偏差，减少 Representation Bias。
4. **参数高效微调**
   - 采用 **LoRA（Low-Rank Adaptation）** 进行高效调优，减少训练资源需求。
5. **模型评估**
   - 采用 **零样本分类（Zero-shot Classification）** 和 **检索任务（Image-Text Retrieval）** 评估泛化能力。
