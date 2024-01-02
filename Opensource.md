# Awesome-Quantization-Papers [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This repo contains a comprehensive paper list of **Model Quantization** for efficient deep learning on AI conferences/journals/arXiv. As a highlight, we categorize the papers in terms of model structures and application scenarios, and label the quantization methods with keywords. `<br>`

This repo is being actively updated, and contributions in any form to make this list more comprehensive are welcome. Special thanks to collaborator [Zhikai Li](https://github.com/zkkli), and all researchers who have contributed to this repo! `<br>`

If you find this repo useful, please consider **★STARing** and feel free to share it with others! `<br>`

**[Update: Nov, 2023]** Add new papers from NeurIPS-23. `<br>`
**[Update: Oct, 2023]** Add new papers from ICCV-23. `<br>`
**[Update: Jul, 2023]** Add new papers from AAAI-23 and ICML-23. `<br>`
**[Update: Jun, 2023]** Add new arXiv papers uploaded in May 2023, especially the hot LLM quantization field. `<br>`
**[Update: Jun, 2023]** Reborn this repo! New style, better experience! `<br>`

---

## Overview

- [Awesome-Quantization-Papers](#awesome-quantization-papers)
  - [Overview](#overview)
  - [Survey](#survey)
  - [Transformer-based Models](#transformer-based-models)
    - [Vision Transformers](#vision-transformers)
    - [Language Transformers](#language-transformers)
  - [Convolutional Neural Networks](#convolutional-neural-networks)
    - [Visual Generation](#visual-generation)
    - [Image Classification](#image-classification)
    - [Other Tasks](#other-tasks)
      - [Object Detection](#object-detection)
      - [Super Resolution](#super-resolution)
      - [Point Cloud](#point-cloud)
  - [References](#references)

**Keywords**: **`PTQ`**: post-training quantization | **`Non-uniform`**: non-uniform quantization | **`MP`**: mixed-precision quantization | **`Extreme`**: binary or ternary quantization

---

## Survey

- "A Survey of Quantization Methods for Efficient Neural Network Inference", Book Chapter: Low-Power Computer Vision, 2021. [[paper](https://arxiv.org/abs/2103.13630)]
- "Full Stack Optimization of Transformer Inference: a Survey", arXiv, 2023. [[paper](https://arxiv.org/abs/2302.14017)]
- "A White Paper on Neural Network Quantization", arXiv, 2021. [[paper](https://arxiv.org/abs/2106.08295)]
- "Binary Neural Networks: A Survey", PR, 2020. [[Paper](https://arxiv.org/abs/2004.03333)] [**`Extreme`**]

## Transformer-based Models

### Vision Transformers

- "I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference", ICCV, 2023. [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_I-ViT_Integer-only_Quantization_for_Efficient_Vision_Transformer_Inference_ICCV_2023_paper.pdf)] [[code](https://github.com/zkkli/I-ViT)]
- "RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers", ICCV, 2023. [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_RepQ-ViT_Scale_Reparameterization_for_Post-Training_Quantization_of_Vision_Transformers_ICCV_2023_paper.pdf)] [[code](https://github.com/zkkli/RepQ-ViT)] [**`PTQ`**]
- "Oscillation-free Quantization for Low-bit Vision Transformers", ICML, 2023. [[paper](https://openreview.net/forum?id=DihXH24AdY)] [[code](https://github.com/nbasyl/OFQ)]
- "Q-ViT: Accurate and Fully Quantized Low-bit Vision Transformer", NeurIPS, 2022. [[paper](https://openreview.net/forum?id=fU-m9kQe0ke)] [[code](https://github.com/yanjingli0202/q-vit)]
- "Patch Similarity Aware Data-Free Quantization for Vision Transformers", ECCV, 2022. [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710154.pdf)] [[code](https://github.com/zkkli/psaq-vit)]  [**`PTQ`**]
- "PTQ4ViT: Post-Training Quantization for Vision Transformers with Twin Uniform Quantization", ECCV, 2022. [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720190.pdf)] [[code](https://github.com/hahnyuan/ptq4vit)]  [**`PTQ`**]
- "FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer", IJCAI, 2022. [[paper](https://arxiv.org/abs/2111.13824)]  [[code](https://github.com/megvii-research/FQ-ViT)]  [**`PTQ`**]

[[Back to Overview](#overview)]

### Language Transformers

- "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS, 2023. [[paper](https://neurips.cc/virtual/2023/poster/71815)] [[code](https://github.com/artidoro/qlora)]
- "QuIP: 2-Bit Quantization of Large Language Models With Guarantees", NeurIPS, 2023. [[paper](https://neurips.cc/virtual/2023/poster/69982)] [[code](https://github.com/jerry-chee/QuIP)] [**`PTQ`**]
- "QIGen: Generating Efficient Kernels for Quantized Inference on Large Language Models", arXiv, 2023. [[paper](https://arxiv.org/abs/2307.03738)] [[code](https://github.com/IST-DASLab/QIGen)]
- "RPTQ: Reorder-based Post-training Quantization for Large Language Models", arXiv, 2023. [[paper](https://arxiv.org/abs/2304.01089)] [[code](https://github.com/hahnyuan/rptq4llm)] [**`PTQ`**]
- "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models", ICML, 2023. [[paper](https://arxiv.org/abs/2211.10438)] [[code](https://github.com/mit-han-lab/smoothquant)] [**`PTQ`**]
- "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", ICLR, 2023. [[papar](https://arxiv.org/abs/2210.17323)]  [[code](https://github.com/IST-DASLab/gptq)] [**`PTQ`**]
- "BiBERT: Accurate Fully Binarized BERT", ICLR, 2022. [[paper](https://openreview.net/forum?id=5xEgrl_5FAJ)] [[code](https://github.com/htqin/BiBERT)] [**`Extreme`**]
- "BiT: Robustly Binarized Multi-distilled Transformer", NeurIPS, 2022. [[paper](https://nips.cc/Conferences/2022/Schedule?showEvent=55032)] [[code](https://github.com/facebookresearch/bit)] [**`Extreme`**]
- "Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Models", NeurIPS, 2022. [[paper]](https://arxiv.org/abs/2209.13325) [[code](https://github.com/wimh966/outlier_suppression)] [**`PTQ`**]
- "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", NeurIPS, 2022. [[paper](https://arxiv.org/abs/2208.07339)] [[code](https://github.com/timdettmers/bitsandbytes)]
- "Towards Efficient Post-training Quantization of Pre-trained Language Models", NeurIPS, 2022. [[paper](https://nips.cc/Conferences/2022/Schedule?showEvent=53407)] [**`PTQ`**]
- "ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers", NeurIPS, 2022. [[paper](https://nips.cc/Conferences/2022/Schedule?showEvent=54407)] [[code](https://github.com/microsoft/DeepSpeed)] [**`PTQ`**]
- "I-BERT: Integer-only BERT Quantization", ICML, 2021. [[paper](https://proceedings.mlr.press/v139/kim21d.html)] [[code](https://github.com/kssteven418/I-BERT)]
- "BinaryBERT: Pushing the Limit of BERT Quantization", ACL, 2021. [[paper](https://arxiv.org/abs/2012.15701)] [[code](https://github.com/huawei-noah/Pretrained-Language-Model)] [**`Extreme`**]
- "Understanding and Overcoming the Challenges of Efficient Transformer Quantization", EMNLP, 2021. [[paper](https://arxiv.org/abs/2109.12948)] [[code](https://github.com/qualcomm-ai-research/transformer-quantization)]
- "TernaryBERT: Distillation-aware Ultra-low Bit BERT", EMNLP, 2020. [[paper](https://arxiv.org/abs/2009.12812)] [[code](https://github.com/huawei-noah/Pretrained-Language-Model)] [**`Extreme`**]
- [[Back to Overview](#overview)]

## References

* Online Resources:
  * [MQBench (Benchmark)](http://mqbench.tech/)
  * [Awesome Model Quantization (GitHub)](https://github.com/htqin/awesome-model-quantization)
  * [Awesome Transformer Attention (GitHub)](https://github.com/cmhungsteve/Awesome-Transformer-Attention)
