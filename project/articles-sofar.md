

https://openreview.net/forum?id=NmNmlAEBAl&nesting=2&sort=date-desc An empirical study of CLIP fine-tuning with similarity clusters









# CLIP the Bias: How Useful is Balancing Data in Multimodal Learning?

https://arxiv.org/abs/2403.04547 CLIP the Bias: How Useful is Balancing Data in Multimodal Learning?

📄 **ICLR 2024 会议论文**
 📍 **作者**: Google DeepMind

------

## **1. 研究背景**

近年来，**多模态学习（Multimodal Learning）** 取得了显著进展，如 **CLIP、BLIP、CoCa** 等模型被广泛应用于 **零样本分类、文本生成、图像检索** 等任务。然而，这些模型存在 **固有的社会偏见（bias）**，可能导致：

- **刻板印象的强化**（如“经理”多被识别为男性、“护士”多被识别为女性）。
- **不公平的模型表现**（如不同性别或种族的识别准确度不同）。
- **多模态数据中的偏差（Bias in multimodal datasets）**，如 **数据采样不均衡、语义关联偏差** 等。

**研究目标**：
 本研究探索 **数据平衡（Data Balancing）** 在缓解 CLIP 偏见上的有效性，并提出 **Multi-Modal Moment Matching (M4)** 方法，从**数据层面**减少 **表示偏差（Representation Bias, RB）和关联偏差（Association Bias, AB）**。

------

## **2. 主要贡献**

✅ **发现 CLIP 存在表示偏差和关联偏差**

- **表示偏差（RB）**：CLIP 可能会将某些属性（如性别、职业）与特定图像过度关联。
- **关联偏差（AB）**：例如，CLIP 可能会将“医生”职业更常与“男性”联系，而非“女性”。

✅ **提出数据平衡算法 Multi-Modal Moment Matching (M4)**

- 通过平衡数据，减少偏差，使 CLIP 在不同属性类别之间更加均衡。
- 研究数据平衡对 **模型泛化能力、零样本分类、检索任务** 等的影响。

✅ **系统性实验：研究数据平衡的影响**

- 训练 **150+ 个模型**，分析不同数据平衡策略的效果。
- 发现数据平衡对 **分类任务有正面影响**，但对 **检索任务可能有负面影响**。

------

## **3. 方法：Multi-Modal Moment Matching (M4)**

📌 **核心思想**：

1. 第一阶统计平衡（Representation Bias, RB）
   - 控制特定类别（如“男性”和“女性”）在数据中的比例，使其符合目标分布（如 50%）。
2. 第二阶统计平衡（Association Bias, AB）
   - 破除不必要的关联（如“职业”与“性别”的强关联）。

📌 **M4 具体实现**：

- **计算样本权重 qq**，使数据分布更均衡：

  L=∑kmax⁡(0,∣E[q⋅(sk−πk)⋅yr]∣−ϵD)+max⁡(0,∣E[q⋅(sk−πk)]∣−ϵR)L = \sum_{k} \max(0, |E[q \cdot (s_k - \pi_k) \cdot y_r]| - ϵ_D) + \max(0, |E[q \cdot (s_k - \pi_k)]| - ϵ_R)

  其中：

  - qq 为数据采样权重；
  - sks_k 为敏感属性（如性别）；
  - yry_r 为类别（如职业）；
  - πk\pi_k 为目标分布；
  - ϵD,ϵRϵ_D, ϵ_R 为控制偏差的超参数。

- **优化目标**：

  - 让数据分布 **更接近均衡目标**，同时 **减少敏感属性与类别之间的强关联**。
  - 通过 **子采样（subsampling）** 方式，在数据层面对比训练进行优化。

------

## **4. 主要实验**

### **(1) 偏差缓解实验**

- **测试 CLIP 在性别、职业等敏感属性上的表现**，观察数据平衡是否有效减少偏差。

- 主要评估数据集：

  - **FairFace, UTKFace, MIAP**（性别偏差）
  - **COCO, Flickr30K**（文本-图像匹配偏差）

- 发现

  ：

  - 数据平衡可以有效降低 **Representation Bias**。

  - 但在 

    Association Bias

     上，效果较为复杂：

    - **加入代理变量（proxies）可能会加剧 AB**，因为它们引入了更多的约束，可能导致冲突。

### **(2) 数据平衡对模型性能的影响**

- **分类任务（Zero-shot Classification）**

  - 评估数据集：**ImageNet, CIFAR100, Caltech101, Birds, Pets**

  - 结果

    ：

    - 数据平衡 **提升** 了模型的分类准确率。

- **检索任务（Image-Text Retrieval）**

  - 评估数据集：**COCO, Flickr30K**

  - 结果

    ：

    - 数据平衡 **降低** 了模型的检索性能（如 COCO R@5 下降）。

- **数据质量 & 模型架构改进**

  - **通过提高数据质量（如去除低质量文本-图像对）**，可以缓解数据平衡对模型性能的负面影响。
  - **使用 SigLIP 架构**（改进版 CLIP），数据平衡不再影响检索任务，且提高了分类精度（ImageNet 0-shot 提升 **77.0% → 77.5%**）。

------

## **5. 主要发现**

✅ **数据平衡可以有效降低表示偏差（RB），但对关联偏差（AB）影响复杂**。
 ✅ **数据平衡提升分类任务表现，但可能降低检索任务性能**。
 ✅ **数据质量和模型架构的改进可以缓解数据平衡的负面影响**。

------

## **6. 启发 & 你的研究如何借鉴？**

1. **数据平衡可以增强 CLIP 语义对齐能力**：
   - 你可以在 **Knowledge-CLIP 训练中结合 M4**，优化 **知识图谱增强的对比学习**。
   - 让 CLIP **更公平地理解不同类别的概念，而不会过度偏向某些类别**。
2. **优化 Hard Negative 采样**：
   - **M4 的数据采样策略可以与 AHNM（Adaptive Hard Negative Mining）结合**。
   - 让 Hard Negatives **不仅更难，还能减少社会偏见**。
3. **结合 SimCLIP 提高数据平衡效果**：
   - SimCLIP 强调 **相似样本簇（Similarity Clusters）**，可以与 M4 结合，提高对比学习的难度，**减少偏见的同时提高泛化能力**。

------

## **7. 总结**

📌 **M4 是一种数据平衡方法，可减少 CLIP 训练中的表示偏差（RB）和关联偏差（AB）**。
 📌 **数据平衡对分类任务有正面影响，但可能降低检索任务性能**，可以通过 **数据质量 & 模型架构优化来缓解**。
 📌 **你的研究可以结合 M4、AHNM、SimCLIP，提高 CLIP 语义对齐 & 数据公平性**。

🚀 **下一步**：你是否希望结合 M4 设计一个 **偏差优化版 Knowledge-CLIP**？我可以帮你制定详细方案！😃



# knowledge Clip

### **Knowledge-CLIP: Contrastive Language-Image Pre-Training with Knowledge Graphs**

📄 **Paper from NeurIPS 2022**
 📍 **Authors**: Xuran Pan, Tianzhu Ye, Dongchen Han, Shiji Song, Gao Huang (Tsinghua University)

------

## **📌 1. 研究背景**

近年来，大规模 **视觉-语言预训练（Vision-Language Pre-training, VLP）** 框架（如 CLIP）取得了显著进展，能够提取多模态表示并适用于下游任务。然而，现有方法主要依赖于 **简单的图像-文本对** 进行训练，**忽略了跨模态的语义关系**，导致：

- **语义理解不足**：模型容易依赖表面 co-occurrence，而不是深层语义对齐。
- **推理能力有限**：在需要推理能力的任务（如视觉问答 VQA）中表现不佳。
- **对抗性鲁棒性较弱**：在涉及细粒度属性（如颜色、关系）时，CLIP 可能会误判。

------

## **📌 2. 主要贡献**

本文提出了一种新的 **知识增强预训练框架（Knowledge-CLIP）**，通过引入 **知识图谱（Knowledge Graph, KG）** 进行预训练，以**提高 CLIP 在语义对齐和推理任务上的表现**。

### ✅ **核心创新**

1. **结合知识图谱（KG）进行 CLIP 预训练**
   - 采用 **多种知识图谱**（如 Visual Genome、ConceptNet、VisualSem），通过实体关系建模增强语义理解。
2. **引入多模态 Transformer 进行跨模态对齐**
   - 在 CLIP 结构的基础上，增加了一个 **跨模态 Transformer（Multi-modal Encoder）**，用于融合实体（图像/文本）和它们的关系信息。
3. **提出新的训练目标**
   - **三元组损失（Triplet-based Loss）**：通过 KG 中的三元组 (h, r, t) 进行训练，提高实体和关系的表示能力。
   - **图结构损失（Graph-based Loss）**：使用 **图神经网络（GNN）** 提取实体之间的结构信息，提升全局理解能力。
   - **知识蒸馏损失（KD Loss）**：通过蒸馏原始 CLIP 模型的知识，防止新任务训练过程中遗忘 CLIP 的原始能力。

------

## **📌 3. 方法概述**

### **(1) 数据准备**

Knowledge-CLIP 采用以下 **三种知识图谱** 进行训练：

- **Visual Genome (VG)**：提供场景图，包含视觉概念的语义关系（如“人-拿着-手机”）。
- **ConceptNet**：一个纯文本的知识图谱，帮助语言层面的关系建模（如“狗是动物”）。
- **VisualSem**：一个多模态知识图谱，包含实体的图片和描述，有助于跨模态对齐。

### **(2) 训练框架**

Knowledge-CLIP 在 CLIP 结构基础上进行了增强：

1. **图像编码器 & 文本编码器**（与 CLIP 相同）：
   - 采用 **Vision Transformer (ViT) 处理图像**
   - 采用 **Transformer-based 文本编码器**
2. **新增：多模态 Transformer**
   - 输入知识图谱中的三元组 **(h, r, t)**
   - **h（头实体）** 和 **t（尾实体）** 可以是图像或文本
   - **r（关系）** 是一个文本 token
   - Transformer 进行特征融合，增强视觉-语言关系对齐。
3. **损失函数**
   - **实体-实体对齐损失（E2E Loss）**：预测实体之间的语义关系。
   - **实体-关系对齐损失（E2R Loss）**：预测实体与关系的匹配程度。
   - **图结构约束损失（G2E Loss）**：利用 GNN 传播关系信息，增强模型对知识结构的理解。

------

## **📌 4. 主要实验结果**

Knowledge-CLIP 在多个 **视觉-语言任务** 上进行了评测，主要对比了 CLIP 以及其他 SOTA 方法：

1. **跨模态检索（Image-Text Retrieval）**

   - 在 **Flickr30K、COCO Caption** 数据集上，Knowledge-CLIP **优于原始 CLIP**，R@1 提高了 **2~3%**。

2. **视觉问答（VQA）**

   - 在 VQA 任务上，Knowledge-CLIP 取得了 **76.1%（test-dev）和 75.2%（test-std）** 的新 SOTA 结果，相比 CLIP **提升 2%+**。

3. **细粒度语义理解（Negation, Color, Position, Size）**

   - 针对某些 

     挑战性语义任务

     （如颜色、方位、大小），Knowledge-CLIP 显著提升了 CLIP 的表现：

     - **颜色识别 +5.7%**
     - **方位关系 +7.1%**
     - **否定句理解 +2.1%**

4. **图像分类（ImageNet）**

   - 相比 CLIP（84.2%），Knowledge-CLIP 提升到 **84.4%**，证明其泛化能力增强。

------

## **📌 5. 关键发现**

1. **注入知识图谱可以提升视觉-语言对齐**
   - 通过引入 KG 关系信息，模型更容易理解 **对象属性、关系**，提升推理能力。
2. **对抗性鲁棒性提高**
   - Knowledge-CLIP 在**颜色、方位、否定句**等方面有更好的表现，说明它能更好地建模细粒度语义。
3. **比 CLIP 训练更高效**
   - 通过 **知识蒸馏（Knowledge Distillation）**，Knowledge-CLIP 只需 **微调（fine-tuning）**，而不是从头训练，计算成本更低。

------

## **📌 6. 未来研究方向**

- **结合更复杂的知识图谱**（如医疗知识图谱、科学知识图谱）
- **拓展到其他多模态任务**（如文本生成、视频理解）
- **优化计算效率**，减少多模态 Transformer 的计算成本

------

## **📌 7. 你的研究如何借鉴 Knowledge-CLIP？**

1. **增强对比学习（Contrastive Learning）**
   - 你可以借鉴 **知识增强 Hard Negative 采样**（在 AHNM 里加入 KG 提取的对比关系）。
2. **改进 Zero-shot Learning**
   - 你可以尝试在 CLIP 训练中 **结合 KG 结构信息**，让模型更好地理解实体之间的关系，提高 zero-shot 分类能力。
3. **Fine-tuning 小数据集**
   - 如果你关注 **如何在小数据集上提升 CLIP 训练效果**，可以使用 Knowledge-CLIP 提供的 **知识蒸馏技术**，让小数据集模型也能学习 KG 中的知识。

------

## **📌 总结**

✅ **Knowledge-CLIP 是 CLIP 的增强版，通过引入知识图谱来提升语义对齐和推理能力。**
 ✅ **相比 CLIP，它在 Image-Text Retrieval、VQA、细粒度理解等任务上表现更优。**
 ✅ **可以借鉴它的思路，在 CLIP 训练中加入知识引导的对比学习，提高模型的结构化推理能力。**

💡 **你如果想结合这个思路，我可以帮你设计一个 Knowledge-Enhanced AHNM 方案！🚀**











https://arxiv.org/abs/2109.05941 : **Efficient Contrastive Learning via Novel Data Augmentation and Curriculum Learning**