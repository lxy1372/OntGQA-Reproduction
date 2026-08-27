# OntGQA Reproduction

本项目复现论文 **Reasoning with Ontology Graph: Toward Type-Constrained Knowledge Graph Question Answering** 中 OntGQA 的核心思路，并在 WebQSP 数据集上完成从类型规划、候选检索、答案判别到生成式回退的完整问答流程。

本项目的目标是复现 OntGQA 的主要方法框架，而非严格复现论文中的最终数值结果。论文作者目前公开了 Freebase Ontology Graph，但尚未公开完整的数据构造、训练和推理代码，因此部分实现细节需要根据论文描述自行完成。

---

## 1. 方法概述

OntGQA 的核心思想是利用 Ontology Graph 对知识图谱问答中的关系搜索空间进行约束。

本项目实现的整体流程为：

```text
Question
   │
   ▼
Planner
预测 Head Type / Tail Type
   │
   ▼
Ontology-constrained Retriever
根据类型约束检索候选答案和证据路径
   │
   ▼
Judge
根据问题、候选答案和证据路径判断候选是否正确
   │
   ├── 存在通过 Judge 的候选
   │        │
   │        ▼
   │     输出答案
   │
   └── 无候选通过
            │
            ▼
         Generator
         生成式回退
```

最终系统主要由以下四个模块组成：

- **Planner**：根据问题预测 ontology type pair。
- **Retriever**：利用 ontology type constraint 从知识图谱中检索候选答案。
- **Judge**：根据问题、候选实体和 reasoning paths 判断候选是否为正确答案。
- **Generator**：当结构化检索流程无法产生有效答案时进行生成式回退。

---

## 2. 项目结构

```text
OntGQA-Reproduction/
├── data/
│   ├── generator/
│   │   └── generator_train.jsonl
│   ├── judge/
│   │   └── judge_train.jsonl
│   ├── ontology/
│   │   └── ontology_graph_freebase.json
│   ├── planner/
│   │   └── planner_train.jsonl
│   └── webqsp/
│       ├── train-00000-of-00002.parquet
│       ├── train-00001-of-00002.parquet
│       ├── validation-00000-of-00001.parquet
│       ├── test-00000-of-00002.parquet
│       └── test-00001-of-00002.parquet
│
├── models/
│   └── Qwen2-1.5B-Instruct/
│
├── outputs/
│   ├── planner_lora/
│   │   └── final/
│   ├── judge_lora/
│   │   └── final/
│   └── generator_lora/
│       └── final/
│
├── results/
│   └── pipeline_test.jsonl
│
├── scripts/
│   ├── build_generator_data.py
│   ├── build_judge_data.py
│   ├── build_planner_data.py
│   ├── train_generator.py
│   ├── train_judge.py
│   ├── train_planner.py
│   ├── evaluate_pipeline_test.py
│   └── run_demo.py
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── ontology.py
│   ├── retriever.py
│   ├── planner_dataset.py
│   ├── planner.py
│   ├── judge_dataset.py
│   ├── judge.py
│   ├── generator_dataset.py
│   ├── generator.py
│   └── pipeline.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 3. 环境

本项目实验环境：

```text
Python        3.10
PyTorch       2.7.0 + CUDA 12.8
Transformers  4.46.3
PEFT          0.13.2
Accelerate    1.0.1
```

主要模型：

```text
Qwen2-1.5B-Instruct
```

训练采用 LoRA：

```text
r              = 8
lora_alpha     = 16
lora_dropout   = 0.05
target_modules = q_proj, v_proj
```

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 4. 数据

### 4.1 WebQSP

实验使用 WebQSP：

```text
Train : 2826 questions
Test  : 1628 questions
```

本项目使用 RoG 处理后的 WebQSP per-question Freebase subgraph 作为问题级知识图谱。

数据文件位于：

```text
data/webqsp/
```

### 4.2 Ontology Graph

Ontology Graph 使用 OntGQA 作者公开的：

```text
ontology_graph_freebase.json
```

文件位于：

```text
data/ontology/ontology_graph_freebase.json
```

本项目统计得到该 ontology graph 包含：

```text
Ontology triples : 32195
Unique relations : 32195
Entity types     : 12369
```

Retriever 根据 ontology 中 relation 的：

```text
Head Type
Relation
Tail Type
```

约束候选关系搜索空间。

---

## 5. 构造训练数据

### 5.1 Planner

```bash
python scripts/build_planner_data.py
```

输出：

```text
data/planner/planner_train.jsonl
```

Planner 根据问题学习预测：

```text
(Head Type, Tail Type)
```

推理时使用 beam search，并保留 Top-3 type pairs。

### 5.2 Judge

```bash
python scripts/build_judge_data.py
```

输出：

```text
data/judge/judge_train.jsonl
```

Judge 输入：

```text
Question
Candidate
Evidence Paths
```

输出：

```text
YES
NO
```

训练数据由 Retriever 产生的候选答案构造，并使用 Gold answer 判断正负样本。

### 5.3 Generator

```bash
python scripts/build_generator_data.py
```

输出：

```text
data/generator/generator_train.jsonl
```

Generator 用于结构化推理无法获得有效答案时的生成式回退。

---

## 6. 模型训练

### 6.1 Planner

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_planner.py
```

最终 LoRA：

```text
outputs/planner_lora/final
```

### 6.2 Judge

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_judge.py
```

最终 LoRA：

```text
outputs/judge_lora/final
```

### 6.3 Generator

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_generator.py
```

最终 LoRA：

```text
outputs/generator_lora/final
```

---

## 7. 推理流程

完整推理流程实现在：

```text
src/pipeline.py
```

主要步骤：

```text
1. Planner 预测 Top-3 ontology type pairs

2. 删除 ontology 中不存在的 type pairs

3. Retriever 根据 type constraint 检索候选答案

4. 合并不同 type pair 得到的重复候选和 reasoning paths

5. Judge 对每个候选进行 YES / NO 判断

6. 根据 Judge margin 对候选进行排序

7. 接受满足 margin threshold 的候选答案

8. 若没有合法候选，则调用 Generator 进行 backoff
```

Judge 使用：

```text
margin = log P(YES) - log P(NO)
```

并使用：

```text
margin threshold = 1.0
```

进行候选筛选。

---

## 8. Demo

运行单问题 Demo：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_demo.py
```

Demo 会展示：

```text
Question
Planner predicted type pairs
Retriever candidates
Evidence paths
Judge scores
Final answers
```

用于观察完整 OntGQA 推理过程。

---

## 9. WebQSP Test 评测

完整测试：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_pipeline_test.py
```

测试集：

```text
WebQSP Test
1628 questions
```

评价指标采用论文中的：

```text
Hit@1
Macro Precision
Macro Recall
Macro F1
```

其中 Hit@1 表示模型排序第一的预测答案是否命中任意 Gold answer。

F1 对每个问题的预测答案集合与 Gold answer 集合计算 Precision、Recall 和 F1，然后对全部问题进行 Macro Average。

---

## 10. 最终结果

本项目在 WebQSP Test 上得到：

```text
Hit@1           : 52.58%
Macro Precision : 46.29%
Macro Recall    : 72.35%
Macro F1        : 49.54%
```

完整预测结果保存在：

```text
results/pipeline_test.jsonl
```

---

## 11. 复现说明

本项目主要目标是复现 OntGQA 的核心方法流程。

当前结果与论文报告结果存在较大差距，主要存在以下实现条件差异。

### 11.1 作者未公开完整实现

OntGQA 作者目前公开了 Freebase Ontology Graph，但未公开完整的：

```text
training code
inference code
data construction scripts
processed training supervision
model checkpoints
```

因此 Planner、Judge 和 Generator 的训练数据需要根据论文描述自行构造。

### 11.2 Planner supervision 无法完全对齐

论文给出了 Planner supervision 的统计数量，但未公开完整的数据构造和过滤规则。

本项目根据：

```text
Gold answer
Shortest reasoning path
Ontology relation signature
```

重新构造 Planner supervision，因此训练数据与作者实际使用的数据可能存在差异。

实验中发现 Planner 是当前系统的重要性能瓶颈。在给定 Oracle type pair 时，Retriever 能够较高比例地恢复 Gold answer，而使用模型预测 type pair 后，Gold recall 会明显下降。

### 11.3 Judge 训练存在实现歧义

论文使用 YES / NO probability 构造 Judge margin：

```text
margin = log P(YES) - log P(NO)
```

但论文给出的 Judge listwise loss 在按公式字面实现时存在优化方向上的歧义。

由于作者未公开 Judge 训练代码，本项目最终采用直接的 YES / NO supervised fine-tuning 训练 Judge，并在推理阶段按照论文定义的 margin 进行候选判断。

### 11.4 Knowledge Graph 不完全一致

论文实验基于 Freebase。

本项目为了在有限计算与存储条件下完成可运行复现，使用 RoG 提供的 WebQSP per-question Freebase subgraph。

因此实际知识图谱环境与论文完整 Freebase 环境并不完全一致。

### 11.5 模型训练方式不同

论文主要实验使用较大语言模型和完整训练设置。

本项目采用：

```text
Qwen2-1.5B-Instruct
+
LoRA
```

完成 Planner、Judge 和 Generator 的训练。

因此本项目结果主要用于验证 OntGQA 方法流程是否能够完整运行，而不将当前结果视为论文数值结果的严格 reproduction。

---

## 12. 当前复现完成内容

本项目已经完成：

```text
✓ WebQSP 数据读取
✓ 官方 Freebase Ontology Graph 解析
✓ Ontology relation indexing
✓ Type-constrained 1-hop retrieval
✓ Type-constrained 2-hop retrieval
✓ Planner supervision 构造
✓ Planner LoRA 训练
✓ Top-3 type pair prediction
✓ Judge supervision 构造
✓ Judge LoRA 训练
✓ Evidence path textualization
✓ YES / NO margin scoring
✓ Generator supervision 构造
✓ Generator LoRA 训练
✓ Generative backoff
✓ 完整 OntGQA Pipeline
✓ WebQSP Test 1628 题完整评测
✓ Hit@1 / Precision / Recall / F1 计算
```

因此当前版本已经能够完整运行：

```text
Question
→ Planner
→ Ontology-constrained Retrieval
→ Judge
→ Generator Backoff
→ Final Answer
```

形成一个完整、可执行的 OntGQA 核心方法复现。

---

## 13. Reference

Paper:

```text
Reasoning with Ontology Graph:
Toward Type-Constrained Knowledge Graph Question Answering
ACL 2026
```

Official OntGQA repository:

```text
https://github.com/shanyongxue/OntGQA
```

RoG:

```text
https://github.com/RManLuo/reasoning-on-graphs
```