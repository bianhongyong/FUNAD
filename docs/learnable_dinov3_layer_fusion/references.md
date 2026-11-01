# 相关工作与文献引用（截至 2026-10-22）

> 对应 [requirements_and_design.md](./requirements_and_design.md)。本表区分“直接相关异常检测文献”“模块设计先例”和“基础视觉特征研究”。这些论文**不是**所提新方法有效性或创新性的证明，需通过消融验证。

## 文献定位矩阵

| ID | 工作 | 与 FUNAD 方案的关系 | 区别/写论文时应注意 |
| --- | --- | --- | --- |
| R1 | DINOv3 (2025) | 冻结 dense visual representation 的基础来源 | 并未证明某个类别的最佳 Transformer 层或可学习融合必然更好 |
| R2 | PatchCore (CVPR 2022) | 正常 Patch reference memory、最近邻异常匹配、Coreset 压缩 | 本项目拟在**可训练的类别条件融合空间**做 NN；注意引用时勿把原始参考库与自训练伪标签库混同 |
| R3 | Dinomaly (CVPR 2025) | 多类别统一异常检测、foundation Transformer 的强相关对照 | 它是重建路径，不是本设计的“原始多层参考库→融合后 NN 残差” |
| R4 | Selective Kernel Networks (CVPR 2019) | Softmax 选择性分支融合和动态加权机制先例 | 不能称“可学习 Softmax 融合”或“条件分支注意力”本身为全新设计 |
| R5 | Adaptively Spatial Feature Fusion / ASFF (2019 arXiv) | 学习不同特征尺度的空间自适应融合；供未来 Patch Gate 参考 | 目标任务是目标检测，与工业异常 NN Reference Fusion 不同 |
| R6 | ALF (ICIP 2026) | 直接讨论图像异常检测中 Transformer 层的可靠性和自适应融合；包含 DINOv3 实验 | ALF 是无可学习参数、基于异常响应 sharpness 的**图像级/层级评分融合**；FUNAD 拟学习类别相关的**特征融合后再做 NN 残差**。需认真讨论该差异；绝不能声称首个 adaptive layer fusion |
| R7 | Feature Pyramid Networks / FPN (CVPR 2017) | 多层视觉特征融合的一般方法论先例 | 与当前跨 Transformer block、类别条件的融合并不等价 |

## 核实过的书目信息与直接链接

### [R1] DINOv3
- Oriane Siméoni et al. **DINOv3.** arXiv:2508.10104, 2025.
- [arXiv 论文页](https://arxiv.org/abs/2508.10104)
- 与本设计相关：大规模自监督 ViT 的 dense features，支撑冻结 backbone 的多个中间层表示可作为研究对象。
- 不做未经验证的推断：不同 block 的异常性能、最优层组合仍应由项目实验确定。

### [R2] PatchCore
- Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, Peter Gehler. **Towards Total Recall in Industrial Anomaly Detection.** CVPR 2022, pp. 14318–14328.
- [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)
- [代码](https://github.com/amazon-research/patchcore-inspection)
- 与本设计相关：normal patch feature memory、Patch-level NN、coreset efficiency。参考库大小增加时需要关注近邻搜索吞吐和存储。

### [R3] Dinomaly
- Jia Guo, Shuai Lu, Weihang Zhang, Fang Chen, Huiqi Li, Hongen Liao. **Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection.** CVPR 2025, pp. 20405–20415.
- [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Dinomaly_The_Less_Is_More_Philosophy_in_Multi-Class_Unsupervised_Anomaly_CVPR_2025_paper.html)
- [预印本](https://arxiv.org/abs/2405.14325)
- 与本设计相关：统一模型处理 MVTec AD 和 VisA 等多个工业类别，适合作为背景、讨论和潜在横向对比。

### [R4] Selective Kernel Networks
- Xiang Li, Wenhai Wang, Xiaolin Hu, Jian Yang. **Selective Kernel Networks.** CVPR 2019, pp. 510–519.
- [CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Selective_Kernel_Networks_CVPR_2019_paper.html)
- [代码](https://github.com/implus/SKNet)
- 与本设计相关：Softmax 分支加权选择。与 Class-Conditioned Layer Fusion 的分支形式相似，但研究对象不同。

### [R5] ASFF
- Songtao Liu, Di Huang, Yunhong Wang. **Learning Spatial Fusion for Single-Shot Object Detection.** arXiv:1911.09516, 2019.
- [arXiv 论文页](https://arxiv.org/abs/1911.09516)
- [代码](https://github.com/ruinmessi/ASFF)
- 与本设计相关：学习空间位置上的不同多尺度分支权重，可供未来 Patch-level 动态层融合的设计参考。
- 说明：文献在此依据已核实的 arXiv 记录引用，不对会议出版状态做未经核实的标注。

### [R6] ALF
- **ALF: Sharpness-Aware Adaptive Layer Fusion for Training-Free Anomaly Detection.** 2026 IEEE International Conference on Image Processing (ICIP), DOI: 10.1109/ICIP61757.2026.11630338.
- [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11630338/)
- [论文提供的代码地址](https://github.com/Deyu008/ALF)（代码是否已公开需单独确认）
- IEEE 文摘明确指出：图像内每层异常图的峰度/集中度引导 reliability weight，不涉及 learned parameters；在 CLIP 与 DINOv3 上评估。它针对的是 anomaly-map layer fusion，而 FUNAD 本需求是特征层的 learned class-conditioned fusion + fused-space NN residual。
- **研究伦理提醒**：这是直接相关的 2026 年已发表工作，在论文 Related Work 中必须对照讨论，不能忽略。

### [R7] FPN
- Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie. **Feature Pyramid Networks for Object Detection.** CVPR 2017, pp. 2117–2125.
- [CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html)
- 与本设计相关：高低层语义与空间信息的融合动机。

## 建议的论文贡献表述（待实验验证）

**避免**：
- “首次在异常检测中使用自适应多层融合。”
- “首次通过 Softmax 学习 Transformer 层选择。”
- “模型自动学会了最优 DINOv3 层。”（没有受控实验不能称最优）
- “全过程端到端可微近邻检索。”（硬 argmin 不可微）

**可以作为待证实的方向**：
1. 在冻结的 DINOv3 上，学习**类别条件**的中间层融合权重，替代人工指定的类别层组合；
2. 将该融合器同时作用于 query 和 raw multi-layer reference，以一致、随训练更新的**融合特征空间**做最近邻，产生 fused residual；
3. 把这一模块嵌入 FUNAD 的自训练伪标签与类别专属 MoE 链路，验证其性能增益、稳定性和适用边界。

## 推荐引用格式（BibTeX 主键示例）

~~~bibtex
@article{simeoni2025dinov3,
  title={DINOv3}, author={Siméoni, Oriane and others},
  journal={arXiv preprint arXiv:2508.10104}, year={2025},
  url={https://arxiv.org/abs/2508.10104}
}
@inproceedings{roth2022patchcore,
  title={Towards Total Recall in Industrial Anomaly Detection},
  author={Roth, Karsten and Pemula, Latha and Zepeda, Joaquin and Schölkopf, Bernhard and Brox, Thomas and Gehler, Peter},
  booktitle={CVPR}, year={2022}, pages={14318--14328}
}
@inproceedings{guo2025dinomaly,
  title={Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection},
  author={Guo, Jia and Lu, Shuai and Zhang, Weihang and Chen, Fang and Li, Huiqi and Liao, Hongen},
  booktitle={CVPR}, year={2025}, pages={20405--20415}
}
@inproceedings{li2019sknet,
  title={Selective Kernel Networks},
  author={Li, Xiang and Wang, Wenhai and Hu, Xiaolin and Yang, Jian},
  booktitle={CVPR}, year={2019}, pages={510--519}
}
@article{liu2019asff,
  title={Learning Spatial Fusion for Single-Shot Object Detection},
  author={Liu, Songtao and Huang, Di and Wang, Yunhong},
  journal={arXiv preprint arXiv:1911.09516}, year={2019}
}
@inproceedings{alf2026icip,
  title={ALF: Sharpness-Aware Adaptive Layer Fusion for Training-Free Anomaly Detection},
  booktitle={IEEE ICIP}, year={2026},
  doi={10.1109/ICIP61757.2026.11630338}
}
@inproceedings{lin2017fpn,
  title={Feature Pyramid Networks for Object Detection},
  author={Lin, Tsung-Yi and Dollár, Piotr and Girshick, Ross and He, Kaiming and Hariharan, Bharath and Belongie, Serge},
  booktitle={CVPR}, year={2017}, pages={2117--2125}
}
~~~

备注：ALF 作者信息本次未从出版元数据逐一核实，BibTeX 暂省略 author 字段，不应编造作者姓名；正式投稿前从 IEEE 导出权威 BibTeX。
