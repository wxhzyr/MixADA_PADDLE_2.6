# MixADA-PADDLE: 通过 Mixup 增强实现更强大的对抗训练

**MixADA-PADDLE** 是论文 [Better Robustness by More Coverage: Adversarial Training with Mixup Augmentation for Robust Fine-tuning](https://arxiv.org/abs/2012.15699) 的 PaddlePaddle 2.6 实现版本。该项目旨在通过结合对抗训练和 Mixup 数据增强技术，提升模型在文本分类任务中的鲁棒性。

[](https://arxiv.org/abs/2012.15699)
[](https://opensource.org/licenses/MIT)

## 简介

自然语言处理 (NLP) 模型的鲁棒性在面对对抗性攻击时至关重要。本项目提出的 **MixADA** (Mixup Adversarial Data Augmentation) 方法，通过在对抗样本上应用 Mixup 技术，生成更多样化、更具挑战性的训练数据，从而在微调阶段显著提升模型的鲁棒性。

此仓库包含了在 PaddlePaddle 2.6 框架下复现论文结果所需的完整代码，包括：

  * **模型定义**: 实现了支持 Mixup 的 BERT 和 RoBERTa 模型 (`MixText`、`SentMix` 等)
  * **训练脚本**: 支持 `nomix`, `tmix`, `sentmix` 等多种训练策略
  * **对抗攻击与评估**: 集成了强大的开源对抗攻击框架 **OpenAttack**，支持多种攻击算法（如 PWWS, TextFooler 等）的评估

## 项目结构

```
.
├── OpenAttack/            # 对抗攻击与评测框架 (PaddlePaddle 版本)
│   ├── attackers/         # 攻击算法实现 (PWWS, TextFooler, etc.)
│   ├── classifiers/       # 模型分类器封装 (支持 Paddle, PyTorch, TensorFlow)
│   └── ...
├── attackEval.py          # 对抗攻击评测脚本
├── mixtext_paddle.py      # MixText 和 SentMix 模型定义
├── run_simMix.py          # 训练脚本 (支持多种 Mixup 策略)
├── run_job.sh             # 示例运行脚本
└── README.md              # 项目说明文档
```

## 核心功能

  * **多种混合策略**:
      * **TMix**: 在词嵌入层面对输入进行线性插值。
      * **SentMix**: 对 [CLS] Token 的表征进行混合。
      * **NoMix**: 标准的对抗训练流程，不使用 Mixup。
  * **灵活的训练流程**:
      * 支持 `BERT` 和 `RoBERTa` 作为基础模型。
      * 可配置混合层 (`--mix_layers_set`) 和混合系数 (`--alpha`)。
      * 支持迭代式对抗训练 (`--iterative`)，在每一轮训练中动态生成对抗样本。
  * **全面的对抗评测**:
      * 基于 **OpenAttack** 框架，提供了标准化的攻击评测流程。
      * 内置多种攻击算法，如 `PWWS`, `TextFooler`, `Genetic` 等。
      * 支持多种评测指标，如攻击成功率、修改率、语义相似度等。

## 环境依赖

本项目在以下环境中测试通过：

  * **PaddlePaddle-GPU**: 2.6.2
  * **PaddleNLP**: \>=2.6.0
  * **Python**: 3.x

您可以通过以下命令安装核心依赖：

```bash
pip install paddlepaddle-gpu==2.6.2 paddlenlp
```

## 数据准备

我们在实验中使用了 **SST-2** 数据集。为方便复现，您可以从以下链接下载处理好的数据：

[**数据下载链接**](https://drive.google.com/file/d/1MIFljjU8sOzxZshBvq7gFqX9MidqUSFe/view?usp=sharing)

下载后，请将数据解压并放置在项目根目录下，或在运行脚本中指定相应路径。

## 快速开始

我们提供了 `run_job.sh` 脚本作为运行示例。在使用前，请根据您的环境修改其中的路径变量，例如 `DATA_PATH` 和 `MODEL_PATH`。

### 1\. 模型训练

以下命令展示了如何使用 **TMixADA** 策略在 **SST-2** 数据集上对 **RoBERTa** 模型进行训练：

```bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PADDLE_HOME=/path/to/your/paddle/home

# 定义路径
DATA_PATH=./sst-2
MODEL_PATH=roberta-base
OUTPUT_PATH=./rbt-sst-tmixada-pwws-iterative

# 运行训练脚本
python run_simMix.py \
    --model_type roberta \
    --mix_type tmix \
    --iterative \
    --attacker pwws \
    --task_name sst-2 \
    --data_dir ${DATA_PATH} \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_PATH} \
    --max_seq_length 128 \
    --mix_layers_set 7 9 12 \
    --alpha 2.0 \
    --num_labels 2 \
    --do_lower_case \
    --per_gpu_train_batch_size 32 \
    --learning_rate 3e-5 \
    --num_train_epochs 5 \
    --logging_steps 200 \
    --eval_all_checkpoints \
    --seed 2020 \
    --overwrite_output_dir \
    --do_train
```

### 2\. 对抗评测

训练完成后，您可以使用 `attackEval.py` 脚本来评估模型的鲁棒性。

```bash
# 评估刚才训练好的模型
python attackEval.py  \
    --model_name_or_path ${OUTPUT_PATH}/best-ep1  \
    --model_type roberta \
    --attacker pwws \
    --data_dir ${DATA_PATH}/test.tsv \
    --max_seq_len 128 \
    --save_dir ./results/pwws_attack.log
```

## OpenAttack 框架

本项目内嵌了 **OpenAttack** 框架的 PaddlePaddle 适配版本。`OpenAttack` 是一个功能强大的文本对抗攻击开源工具包，其主要模块包括：

  * **Attacker** (`OpenAttack/attackers`): 实现了多种经典的攻击算法。
  * **Classifier** (`OpenAttack/classifiers`): 提供了对不同深度学习框架（PaddlePaddle, PyTorch, TensorFlow）模型的统一封装。
  * **AttackEval** (`OpenAttack/attack_evals`): 用于评测攻击效果和模型鲁棒性的核心模块。
  * **Substitute** (`OpenAttack/substitutes`): 提供基于不同策略的词语替换方法，如基于 WordNet 或词向量的替换。
  * **DataManager**: 自动管理和下载所需资源（如预训练模型、词向量等）。

## 引用

如果您觉得我们的工作对您的研究有所帮助，请考虑引用以下论文：

```
@inproceedings{Si2020BetterRB,
  title={Better Robustness by More Coverage: Adversarial Training with Mixup Augmentation for Robust Fine-tuning},
  author={Chenglei Si and Zhengyan Zhang and Fanchao Qi and Zhiyuan Liu and Yasheng Wang and Qun Liu and Maosong Sun},
  booktitle={Findings of ACL},
  year={2021},
}
```

## 联系我们

如果您在使用过程中遇到任何问题，欢迎通过 GitHub Issues 或直接联系作者进行交流。
