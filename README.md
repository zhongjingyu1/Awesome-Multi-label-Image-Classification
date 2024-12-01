# Awesome-Multi-label-image-classification
We've compiled a list of related work in this not-so-popular multi-label field. We hope it will be helpful to the researchers involved.
## 3. Benchmark Datasets
|  Dataset   | # Class  | # Training data	| # Test data| #Average labels per image|
|  ----  | ----  |----  |----  |----|
|[Pascal VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/) (`VOC07`)|20|5,011|4952|2.5|
|[Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/) (`VOC12`)|20|11,540|10,991|2.5|
|[MS-COCO 2014](https://cocodataset.org/#download) (`COCO`)|80|82,081|40,504|2.9|
|MS-COCO 2017 (`COCO17`)|80|118,287|40,670|-|
|NUS-WIDE (`NW`)|81|125,449|83898|2.4|
|WIDER Attribute (`WA`)|14|28,345|29,179||
|PA-100K (`PA`)|26|9,000|1,000||
|Fashion550K (`F550`)|66||||
|Charades (`Cha`)|157|8,000|1,800|6.8|
|Visual Genome (`VG500`)|500|75,774|32,475|-|
|Visual Genome 256 (`VG256`)|256|75,774|32,475|-|
|IAPRTC-12 (`IA12`)|275|13,989|6,011|-|
## 4. Top-tier Conference Papers
### 2014--2019
|  Title   | Venue  | Year| Datasets | Code|
|  ----  | ----  |----  |----  |----  |
|**[WARP]**[Deep Convolutional Ranking for Multilabel Image Annotation](https://arxiv.org/pdf/1312.4894)|-|2014|`NW`|-|
|**[HCP]**[HCP: AFlexible CNN Framework for Multi-Label Image Classification](http://mepro.bjtu.edu.cn/res/papers/2016/HCP_%20A%20Flexible%20CNN%20Framework%20for%20Multi-Label%20Image%20Classification.pdf)|TPAMI|2016|`VOC07`,`VOC12`|-|
|**[CNN-RNN]**[CNN-RNN: AUnified Framework for Multi-label Image Classification](https://openaccess.thecvf.com/content_cvpr_2016/papers/Wang_CNN-RNN_A_Unified_CVPR_2016_paper.pdf)|CVPR|2016|`NW`,`COCO`,`VOC07`|[Official](https://github.com/shemayon/CNN-RNN-A-Unified-Framework-for-Multi-Label-Image-Classification)|
|**[FeV+LV]**[Exploit Bounding Box Annotations for Multi-Label Object Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yang_Exploit_Bounding_Box_CVPR_2016_paper.pdf)|CVPR|2016|`VOC07`,`VOC12`|-|
|**[RCP]**[Beyond Object Proposals: Random Crop Pooling for Multi-Label Image Recognition](https://ieeexplore.ieee.org/abstract/document/7574320)|TIP|2016|`VOC07`,`VOC12`|-|
|**[SRN]**[Learning Spatial Regularization with Image-level Supervisions for Multi-label Image Classification](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhu_Learning_Spatial_Regularization_CVPR_2017_paper.pdf)|CVPR|2017|`NW`,`COCO`|[Official](https://github.com/zhufengx/SRN_multilabel/)|
|**[LSEP]**[Improving Pairwise Ranking for Multi-label Image Classification](https://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Improving_Pairwise_Ranking_CVPR_2017_paper.pdf)|CVPR|2017|`NW`,`COCO`,`VOC07`|[Official](https://github.com/shjo-april/Tensorflow_Improving_Pairwise_Ranking_for_Multi-label_Image_Classification)|
|**[-]**[Order-Free RNN with Visual Attention for Multi-Label Classification](https://arxiv.org/pdf/1707.05495v1)|AAAI|2017|`NW`,`COCO`|-|
|**[Multi-Evidence]**[Multi-Evidence Filtering and Fusion for Multi-Label Classification, Object Detection and Semantic Segmentation Based on Weakly Supervised Learning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Ge_Multi-Evidence_Filtering_and_CVPR_2018_paper.pdf)|CVPR|2017|`COCO`,`VOC07`,`VOC12`|-|
|**[RDAR]**[Multi-label Image Recognition by Recurrently Discovering Attentional Regions](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wang_Multi-Label_Image_Recognition_ICCV_2017_paper.pdf)|ICCV|2017|`COCO`,`VOC07`|[Official](https://github.com/James-Yip/AttentionImageClass)|
|**[KD-WSD]**[Multi-Label Image Classification via Knowledge Distillation from Weakly-Supervised Detection](https://arxiv.org/pdf/1809.05884)|ACM MM|2018|`NW`,`COCO`|[Official](https://github.com/Yochengliu/MLIC-KD-WSD)|
|**[RLSD]**[Multilabel Image Classification With Regional Latent Semantic Dependencies](https://click.endnote.com/viewer?doi=10.1109%2Ftmm.2018.2812605&token=WzM0ODAwNzgsIjEwLjExMDkvdG1tLjIwMTguMjgxMjYwNSJd.GJxNPnwI4HLO2_AbRR2Bw0ha3jM)|TMM|2018|`NW`,`COCO`,`VOC07`|-|
|**[RARL]**[Recurrent Attentional Reinforcement Learning for Multi-label Image Recognition](https://ojs.aaai.org/index.php/AAAI/article/view/12281/12140)|AAAI|2018|`COCO`,`VOC07`|-|
|**[RMIC]**[Reinforced Multi-label Image Classification by Exploring Curriculum](https://ojs.aaai.org/index.php/AAAI/article/view/11770/11629)|AAAI|2018|`VOC07`,`VOC12`|-|
|**[MsDPD]**[Multi-scale and Discriminative Part Detectors Based Features for Multi-label Image Classification](https://www.ijcai.org/Proceedings/2018/0090.pdf)|IJCAI|2018|`VOC07`,`VOC12`|-|
|**[ML-GCN]**[Multi-label Image Recognition With Graph Convolutional Networks](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Multi-Label_Image_Recognition_With_Graph_Convolutional_Networks_CVPR_2019_paper.pdf)|CVPR|2019|`COCO`,`VOC07`|[Official](https://github.com/megvii-research/ML-GCN)|
|**[SSGRL]**[Learning Semantic-specific Graph Representation for Multi-label Image Recognition](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Learning_Semantic-Specific_Graph_Representation_for_Multi-Label_Image_Recognition_ICCV_2019_paper.pdf)|ICCV|2019|`COCO`,`VOC07`,`VOC12`|[Official](https://github.com/HCPLab-SYSU/SSGRL)|
|**[CADM]**[Multi-label Image Recognition with Joint Class-aware Map Disentangling and Label Correlation Embedding](https://www.lamda.nju.edu.cn/weixs/publication/icme19.pdf)|ICME|2019|`NW`,`COCO`|-|
|**[ACfs]**[Visual Attention Consistency under Image Transforms for Multi-Label Image Classification](https://openaccess.thecvf.com/content_CVPR_2019/papers/Guo_Visual_Attention_Consistency_Under_Image_Transforms_for_Multi-Label_Image_Classification_CVPR_2019_paper.pdf)|CVPR|2019|`COCO`,`WA`,`PA`|[Official](https://github.com/hguosc/visual_attention_consistency)|
|**[CoP]**[Multilabel Image Classification via Feature/Label Co-Projection](https://ieeexplore.ieee.org/document/8985434)|TSMC|2019|`COCO`,`VOC07`|-|
|**[DELTA]**[DELTA: A deep dual-stream network for multi-label image classification](https://www.sciencedirect.com/science/article/pii/S0031320319301050)|PR|2019|`COCO`,`VOC07`|-|

### 2020
|  Title   | Venue  | Year| Datasets | Code|
|  ----  | ----  |----  |----  |----  |
|**[PLA]**[Orderless Recurrent Models for Multi-label Classification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yazici_Orderless_Recurrent_Models_for_Multi-Label_Classification_CVPR_2020_paper.pdf)|CVPR|2020|`NW`,`COCO`,`WA`,`PA`|[Official](https://github.com/voyazici/orderless-rnn-classification)|
|**[TSGCN]**[Joint Input and Output Space Learning for Multi-Label Image Classification](https://ieeexplore.ieee.org/document/9115821)|TMM|2020|`COCO`,`VOC07`|-|
|**[CMA]**[Cross-Modality Attention with Semantic Graph Embedding for Multi-Label Classification](https://ojs.aaai.org/index.php/AAAI/article/view/6964/6818)|AAAI|2020|`NW`,`COCO`|-|
|**[KSSNet]**[Multi-Label Classification with Label Graph Superimposing](https://arxiv.org/pdf/1911.09243)|AAAI|2020|`COCO`,`Cha`|[Official](https://github.com/mathkey/mssnet)|
|**[A-GCN]**[Learning Class Correlations for Multi-label Image Recognition with Graph Networks](https://www.sciencedirect.com/science/article/pii/S0167865520302968)|PRL|2020|`COCO`,`F550`|[Official](https://github.com/queenie88/A-GCN)|
|**[ADD-GCN]**[Attention-Driven Dynamic Graph Convolutional Network for Multi-label Image Recognition](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660647.pdf)|ECCV|2020|`COCO`,`VOC07`,`VOC12`|[Official](https://github.com/Yejin0111/ADD-GCN)|
|**[F-GCN]**[Fast Graph Convolution Network Based Multi-label Image Recognition via Cross-modal Fusion](https://dl.acm.org/doi/10.1145/3340531.3411880)|CIKM|2020|`COCO`,`VOC07`|-|
|**[MSSAF]**[Multi-scale Cross-modal Spatial Attention Fusion for Multi-label Image Recognition](https://link.springer.com/content/pdf/10.1007/978-3-030-61609-0_58.pdf)|ICANN|2020||-|

### 2021
|  Title   | Venue  | Year| Datasets | Code|
|  ----  | ----  |----  |----  |----  |
|**[MCAR]**[Learning to Discover Multi-Class Attentional Regions for Multi Label Image Recognition](https://ieeexplore.ieee.org/document/9466402)|TIP|2021|`COCO`,`VOC07`,`VOC12`|[Official](https://github.com/gaobb/MCAR)|
|**[GM-MLIC]**[GM-MLIC: Graph Matching based Multi-Label Image Classification](https://www.ijcai.org/proceedings/2021/0163.pdf)|IJCAI|2021|`NW`,`COCO`,`VOC07`|-|
|**[MGTN]**[Modular Graph Transformer Networks for Multi-Label Image Classification](https://ojs.aaai.org/index.php/AAAI/article/view/17098/16905)|AAAI|2021|`COCO`,`F550`|[Official](https://github.com/ReML-AI/MGTN)|
|**[DSDL]**[Deep Semantic Dictionary Learning for Multi-label Image Classification](https://ojs.aaai.org/index.php/AAAI/article/download/16472/16279)|AAAI|2021|`COCO`,`VOC07`,`VOC12`|[Official](https://github.com/FT-ZHOU-ZZZ/DSDL)|
|**[Q2L]**[Query2Label: A Simple Transformer Way to Multi-Label Classification](https://arxiv.org/pdf/2107.10834v1)|arXiv|2021|`NW`,`COCO`,`VOC07`,`VOC12`,`VG500`|[Official](https://github.com/SlongLiu/query2labels)|
|**[C-Tran]**[General Multi-label Image Classification with Transformers](https://openaccess.thecvf.com/content/CVPR2021/papers/Lanchantin_General_Multi-Label_Image_Classification_With_Transformers_CVPR_2021_paper.pdf)|CVPR|2021|`COCO`,`VG500`|[Official](https://github.com/QData/C-Tran)|
|**[CSRA]**[Residual Attention: A Simple But Effective Method for Multi-Label Recoginition](https://arxiv.org/pdf/2108.02456v1)|ICCV|2021|`COCO`,`VOC07`,`VOC12`,`WA`|[Official](https://github.com/Kevinz-code/CSRA)|
|**[ASL]**[Asymmetric Loss For Multi-Label Classification](https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf)|ICCV|2021|`NW`,`COCO`,`VOC07`|[Official](https://github.com/Alibaba-MIIL/ASL)|
|**[TDRG]**[Transformer-based Dual Relation Graph for Multi-label Image Recognition](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Transformer-Based_Dual_Relation_Graph_for_Multi-Label_Image_Recognition_ICCV_2021_paper.pdf)|ICCV|2021|`COCO`,`VOC07`|[Official](https://github.com/iCVTEAM/TDRG)|
|**[TkML-AP]**[TkML-AP: Adversarial Attacks to Top-k Multi-Label Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Hu_TkML-AP_Adversarial_Attacks_to_Top-k_Multi-Label_Learning_ICCV_2021_paper.pdf)|ICCV|2021|`COCO`,`VOC12`|[Official](https://github.com/discovershu/TKML-AP)|
|**[M3TR]**[M3TR: Multi-modal Multi-label Recog nition with Transformer](https://dl.acm.org/doi/pdf/10.1145/3474085.3475191)|ACM MM|2021|`COCO`,`VOC07`|[Official](https://github.com/iCVTEAM/M3TR)|
|**[AdaHGNN]**[AdaHGNN: Adaptive Hypergraph Neural Networks for Multi-Label Image Classification](https://openreview.net/pdf?id=pKCf8staKu)|ACM MM|2021|`NW`,`COCO`,`VOC07`,`VG500`|-|
|**[Deep-LIFT]**[Deep-LIFT: Deep Label-Specific Feature Learning for Image Annotation](https://ieeexplore.ieee.org/document/9352498)|TCYB|2021|`COCO`,`VOC07`|[Official](https://github.com/yidaiqiushen/Deep-LIFT)|
|**[DRTE]**[Distance Restricted Transformer Encoder for Multi-Label Classification](https://ieeexplore.ieee.org/document/9428164)|ICME|2021|`COCO`,`WA`|[Official](https://github.com/xiaomeiyy/DRTE)|

### 2022
|  Title   | Venue  | Year| Datasets | Code|
|  ----  | ----  |----  |----  |----  |
|**[CPSD]**[Boosting Multi-Label Image Classification with Complementary Parallel Self-Distillation](https://www.ijcai.org/proceedings/2022/0208.pdf)|IJCAI|2022|`NW`,`COCO`|[Official](https://github.com/Robbie-Xu/CPSD)|
|**[]**[Global Meets Local: Effective Multi-Label Image Classification via Category-Aware Weak Supervision](https://arxiv.org/pdf/2211.12716)|ACM MM|2022|`COCO`,`VOC07`|-|
|**[TSFormer]**[Two-Stream Transformer for Multi-Label Image Classification](https://dl.acm.org/doi/pdf/10.1145/3503161.3548343)|ACM MM|2022|`NW`,`COCO`,`VOC07`|[Official](https://github.com/jasonseu/TSFormer#two-stream-transformer-for-multi-label-image-classification)|
|**[]**[Label-aware Graph Representation Learning for Multi-label Image Classification](https://www.sciencedirect.com/science/article/pii/S0925231222003721)|Neurocomputing|2022|`COCO`,`VOC07`,`VOC12`|-|
|**[GCN-MS-SGA]**[A Multi-scale Semantic Attention Representation for Multi-label Image Recognition with Graph Networks](https://www.sciencedirect.com/science/article/pii/S0925231222003526)|Neurocomputing|2022|`COCO`,`VOC07`|-|
|**[HGT&BC]**[Hierarchical GAN-Tree and Bi-Directional Capsules for Multi-label Image Classification](https://www.sciencedirect.com/science/article/pii/S0950705121010510)|KBS|2022|`NW`,`COCO`,`VOC07`,`IA12`|-|
|**[SSNP]**[Semantic Supplementary Network With Prior Information for Multi-Label Image Classification](https://ieeexplore.ieee.org/abstract/document/9441021)|TCSVT|2022|`NW`,`COCO`,`VOC07`|-|
|**[CPCL]**[Multi-Label Image Classification via Category Prototype Compositional Learning](https://ieeexplore.ieee.org/abstract/document/9614128)|TCSVT|2022|`NW`,`COCO`,`VOC07`|[Official](https://github.com/FT-ZHOU-ZZZ/CPCL)|
|**[SST]**[SST: Spatial and Semantic Transformers for Multi-Label Image Recognition](https://ieeexplore.ieee.org/document/9733172)|TIP|2022|`NW`,`COCO`,`VOC07`|-|
|**[KGGR]**[Knowledge-Guided Multi-Label Few-Shot Learning for General Image Recognition](https://ieeexplore.ieee.org/abstract/document/9207855)|TPAMI|2022|`COCO`,`VOC07`,`VOC12`,`VG500`|-|
|**[MLTR]**[MLTR: Multi-Label Classification with Transformer](https://ieeexplore.ieee.org/abstract/document/9860016)|ICME|2022|`NW`,`COCO`,`VOC07`|-|

### 2023
|  Title   | Venue  | Year| Datasets | Code|
|  ----  | ----  |----  |----  |----  |
|**[]**[MuMIC-Multimodal Embedding for Multi-Label Image Classification with Tempered Sigmoid](https://ojs.aaai.org/index.php/AAAI/article/view/26850/26622)|AAAI|2023|-|-|
|**[]**[Open-Vocabulary Multi-Label Classification via Multi-Modal Knowledge Transfer](https://ojs.aaai.org/index.php/AAAI/article/view/25159/24931)|AAAI|2023|`NW`,`open`|[Official](https://github.com/sunanhe/MKT)|
|**[CDUL]**[CDUL: CLIP-Driven Unsupervised Learning for Multi-Label Image Classification](https://openaccess.thecvf.com/content/ICCV2023/papers/Abdelfattah_CDUL_CLIP-Driven_Unsupervised_Learning_for_Multi-Label_Image_Classification_ICCV_2023_paper.pdf)|ICCV|2023|`NW`,`COCO`,`VOC07`,`VOC12`|[Official](https://github.com/cs-mshah/CDUL)|
|**[PatchCT]**[PatchCT: Aligning Patch Set and Label Set with Conditional Transport for Multi-Label Image Classification](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_PatchCT_Aligning_Patch_Set_and_Label_Set_with_Conditional_Transport_ICCV_2023_paper.pdf)|ICCV|2023|`NW`,`COCO`,`VOC07`|-|
|**[SALGL]**[Scene-Aware Label Graph Learning for Multi-Label Image Classification](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Scene-Aware_Label_Graph_Learning_for_Multi-Label_Image_Classification_ICCV_2023_paper.pdf)|ICCV|2023|`NW`,`COCO`,`VOC07`|[Official](https://github.com/jasonseu/SALGL/tree/main)|
|**[HSVLT]**[HSVLT: Hierarchical Scale-Aware Vision-Language Transformer for Multi-Label Image Classification](https://arxiv.org/pdf/2407.16244)|ACM MM|2023|`NW`,`COCO`,`VOC07`|[Official](https://github.com/NaturalKnight/HSVLT)|
|**[SCO-DCNN]**[Spatial Context-Aware Object-Attentional Network for Multi-Label Image Classification](https://ieeexplore.ieee.org/document/10122681)|TIP|2023|`COCO17`,`VOC07`|-|
|**[VSGCN]**[Beyond Word Embeddings: Heterogeneous Prior Knowledge Driven Multi-Label Image Classification](https://ieeexplore.ieee.org/abstract/document/9765340)|TMM|2023|`NW`,`COCO`,`VOC07`|-|
|**[P-GCN]**[Learning Graph Convolutional Networks for Multi-Label Recognition and Applications](https://ieeexplore.ieee.org/document/9369105)|TPAMI|2023|`NW`,`COCO`,`VOC07`|-|
|**[MulCon]**[Contrastively enforcing distinctiveness for multi-label image classification](https://www.sciencedirect.com/science/article/pii/S0925231223007282)|Neurocomputing|2023|`NW`,`COCO`,`VOC07`,`VG500`|[Official](https://github.com/SonDaoDuy/MulCon/tree/main)|
|**[]**[Balanced Masking Strategy for Multi-label Image Classification](https://www.sciencedirect.com/science/article/pii/S0925231222014023)|Neurocomputing|2023|`COCO`,`VOC07`|-|
|**[FL-Tran]**[Feature Learning Network with Transformer for Multi-label Image Classification](https://www.sciencedirect.com/science/article/pii/S0031320322006823)|PR|2023|`NW`,`COCO`,`VOC07`|-|
|**[ML-Decoder]**[ML-Decoder: Scalable and Versatile Classification Head](https://openaccess.thecvf.com/content/WACV2023/papers/Ridnik_ML-Decoder_Scalable_and_Versatile_Classification_Head_WACV_2023_paper.pdf)|WACV|2023|`NW`,`COCO`,`VOC07`|[Official](https://github.com/Alibaba-MIIL/ML_Decoder)|

### 2024
|  Title   | Venue  | Year| Datasets | Code|
|  ----  | ----  |----  |----  |----  |
|**[HSQ]**[Hybrid Sharing for Multi-Label Image Classification](https://openreview.net/pdf?id=yVJd8lKyVX)|ICLR|2024|`COCO`,`VOC07`|[Official](https://github.com/zihao-yin/HSQ)|
|**[MLC-PAT]**[Counterfactual Reasoning for Multi-Label Image Classification via Patching-Based Training](https://openreview.net/attachment?id=lQIN9ZyMLz&name=pdf)|ICML|2024|`COCO`,`VOC07`,`VG256`|[Official](https://github.com/xiemk/MLC-PAT)|
|**[DRGN]**[Mining Semantic Information With Dual Relation Graph Network for Multi-Label Image Classification](https://ieeexplore.ieee.org/document/10128751)|TMM|2024|`COCO`,`VOC07`,`VG500`|-|
|**[CCR]**[Conditional Consistency Regularization for Semi-Supervised Multi-Label Image Classification](https://ieeexplore.ieee.org/abstract/document/10283961)|TMM|2024|`COCO`,`VOC07`|-|
|**[DATran]**[DATran: Dual Attention Transformer for Multi-Label Image Classification](https://ieeexplore.ieee.org/document/10147882)|TCSVT|2024|`COCO`,`VOC07`,`VG500`|-|
|**[C-TMS]**[Transformer Driven Matching Selection Mechanism for Multi-Label Image Classification](https://ieeexplore.ieee.org/document/10158710)|TCSVT|2024|`NW`,`COCO`,`VOC07`,`VOC12`|[Official]()|
|[]()||||[Official]()|

Attentive recurrent neural network for weak-supervised multi-label image classification

## 5. Other Resources
- [Awesome-Multi-label-Image-Recognition](https://github.com/JiaweiZhao-git/Awesome-Multi-label-Image-Recognition)
- [Awesome-Multi-label-Image-Recognition-1](https://github.com/zhouchunpong/Awesome-Multi-label-Image-Recognition-1)
