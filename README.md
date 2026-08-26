# OntGQA-Reproduction

**OntGQA: Reasoning with Ontology Graph: Toward Type-Constrained Knowledge Graph Question Answering** 论文核心推理流程的轻量级复现。

## 当前目标

第一阶段目标是在 5–20 个 WebQSP 样本上复现 OntGQA 核心推理流程，而非完整复现原论文的全部性能。

该流程将包括：

1. 本体引导的类型规划（Ontology-guided type planning）
2. 类型约束的 1-hop / 2-hop 路径检索（Type-constrained path retrieval）
3. 候选答案与证据路径构建（Candidate answer and evidence path construction）
4. 基于大语言模型的答案判断（LLM-based answer judging）
5. 生成式回退（Generative backoff）

## 当前进度

- [x] 项目初始化
- [ ] 检查官方 Freebase 本体图结构
- [ ] 准备 WebQSP 子图数据
- [ ] 实现本体索引模块
- [ ] 实现 1-hop / 2-hop 检索模块
- [ ] 实现规划器（Planner）
- [ ] 实现判断器（Judge）
- [ ] 实现生成式回退模块（Generative backoff）
- [ ] 在 5–20 个 WebQSP 样本上运行演示