<div align="center">

# 曹雨佳 · caoyujia

### AI Infra / Backend / Cloud Native

专注于大模型推理系统、算力基础设施与后端工程。

[个人主页](https://world-cyj.github.io/) · [GitHub](https://github.com/world-cyj) · [Email](mailto:cyj2582329754@gmail.com)

</div>

---

## 关于我

你好，我是曹雨佳，目前在深圳大学攻读计算机科学与技术硕士。

我的工作和学习主要围绕 AI Infra 展开：从后端服务、资源管理和可观测性，到 vLLM / SGLang 推理引擎、KV Cache 传输、PD 分离、GPU Kernel 与多设备迁移。我更关心系统在真实负载下如何运行、如何定位瓶颈，以及如何把一次实验沉淀成可复用的工程能力。

`Backend` → `AI Systems` → `GPU / NPU Infrastructure`

## 当前关注

- **LLM Inference**：vLLM、SGLang、TensorRT-LLM、MiniVLLM、Continuous Batching、PagedAttention。
- **PD 分离与 KV Cache**：Prefill / Decode 解耦、P2P 传输、Layerwise Cache、Prefix Cache、冷热迁移。
- **推理传输链路**：Mooncake、LMCache、NIXL、PyNCCL Connector、RDMA / RoCE、D2D / H2D / D2H。
- **GPU 与性能工程**：CUDA Graph、CUTLASS、GEMM、FlashMLA、DeepGEMM、NCCL-tests、nsys / ncu。
- **模型部署与评测**：DeepSeek-R1、DeepSeek-V4、Qwen3-30B-A3B-FP8、Qwen3-32B、KTransformers。
- **异构算力基础设施**：Ascend 910B3、FlexNPU、多卡推理、热迁移恢复、设备侧缓存与传输。
- **后端与平台**：Go、C++、Python、Kubernetes、Docker、MySQL、Prometheus、Grafana。

## 工作经历

### 百度国际科技（深圳）有限公司

**IaaS 研发实习生（后台功能开发） · 分布式云边缘计算组**  
`2026.05 – 至今`

- 参与 AICP 算力平台与 OPS 运营平台建设，面向分布式云边缘算力交付、集群准入、运营排障和资源管理场景开发后台能力。
- 负责 Go 后端接口、DAO / Service 逻辑、OPS 页面联调、debug / sandbox 验证和 CR 修改。
- 补齐智算中心、实例管理、集群服务、白名单配额、准入配置、审计日志等功能；开发知识库表访问代理接口，支撑跨集群查询、问题定位和验收数据回溯。
- 设计并开发千帆验收智能体本地函数化能力，将部分智能体验收流程迁移到 AICP 服务侧，完成配置、自测、提测和验收文档。
- 主导 `aicp_cutlass_bench` GPU Benchmark 插件开发，基于 CUTLASS 搭建可现场 `nvcc` 编译、覆盖 INT8 / FP8 / FP16 / TF32 / FP32 的 GEMM 测试链路。
- 在 RTX 4090 8 卡环境中，CUTLASS INT8 / FP8 / FP16 相对原 Benchmark 分别提升 `4.81% / 2.59% / 7.80%`；H100 FP32 从 `40.2` 提升至 `49.8 TFLOPS`，提升 `23.7%`。

### 深圳华为云计算技术有限公司

**AI Infra / 后端开发实习生 · 云业务架构与设计部（ICT BG）**  
`2025.03 – 2026.01`

- 参与 FlexNPU 大模型推理热迁移与传输链路研发，面向 Ascend 910B3 多卡推理任务的业务无感迁移和恢复。
- 关注 PD 分离下的 KV Cache 流转、D2D 数据搬运、长上下文缓存管理以及多设备推理任务的资源状态。
- 参与 KV Cache 迁移流程验证、通信域初始化优化、D2D 正确性测试和多卡恢复瓶颈排查。
- 维护缓存块元信息、传输任务生命周期、状态同步、超时恢复和资源释放相关后端逻辑。
- 跑通 `0-3` 卡到 `4-7` 卡的迁移验证；将 `0-7` 卡通信域拆分为 4 个 2 卡通信域后，restore 最大约 `4.5s`。
- 记录 DeepSeek-R1-Distill-Llama-8B 单卡 restore `2.6s`、QwQ-32B 四卡 restore `10.2s`；单机 8 个单卡推理 Pod 在 8 / 64 并发下平均吞吐提升 `0.7% / 2.2%`。

## 学习与研究目录

### 1. PD 分离与传输

- Mooncake 技术调研与源码阅读
- vLLM 基于 P2P / NCCL 的 PD 分离
- SGLang + Mooncake PD 分离部署
- P2P KV Cache 传输设计与实现
- PyNCCL Connector / StatelessProcessGroup 测试
- RDMA、RoCE、HCCL、NCCL 与 D2D 传输
- Layerwise 多级缓存策略与 Cache Reuse

### 2. vLLM

- vLLM v1 推理框架与调度路径
- vLLM P2P KV Cache 传输
- vLLM + LMCache 性能评测
- FlashMLA 性能验证
- Prefix Cache / Decode Cache Reuse
- Chunked Prefill、Scheduler 与 BlockManager
- vLLM 源码边界、Connector 生命周期和异常恢复

### 3. SGLang

- SGLang 框架结构与使用方式
- SGLang 内存管理技术
- Chunked Prefill 与 CUDA Graph
- CUDA Kernel 优化原理与 SGLang 实现机制
- SGLang Router、调度器与传输引擎适配
- SGLang 源码阅读、目录整理和源码分享计划
- SGLang PD 分离与 P2P 传输性能验证

### 4. GPU 与性能工程

- GPU 矩阵计算与 GEMM 性能测试
- CUTLASS TensorOp / SIMT 双路径
- Tile Size、Pipeline Stage、Cluster、Swizzle 调优
- CUDA Kernel、CUDA Event 与 Kernel 延迟
- CUDA Graph 性能验证
- NCCL-tests 多机多卡测试
- nsys / ncu 性能分析与压测方法

### 5. 模型与推理实践

- DeepSeek-R1 系列模型在 RTX 4090 环境下的部署和性能观察
- DeepSeek-V4 性能资料整理与推理路径跟踪
- Qwen3-30B-A3B-FP8 的 1P1D / PD 分离压测
- Qwen3-32B 双 GPU 部署与 Mini-vLLM 优化
- KTransformers 部署 DeepSeek-R1
- TensorRT-LLM 技术学习
- FlashAttention、FlashMLA、DeepEP、DeepGEMM 与 EPLB

### 6. NPU 与异构基础设施

- FlexNPU 大模型推理热迁移
- Ascend 910B3 多卡推理与恢复
- Device / Host / External Cache 分层
- 通信域初始化、状态同步与超时恢复
- 业务无感迁移、弹性伸缩和高可用推理

## 精选项目

### [MinivLLM_plus](https://github.com/world-cyj/MinivLLM_plus)

个人公开项目。围绕 Mini-vLLM 的推理流程进行学习与实现，关注 Scheduler、BlockManager、PagedAttention、Prefix Cache、Decode Kernel 和 Qwen3-32B 部署。

### [AI-System-Performance-Lab](https://github.com/world-cyj/AI-System-Performance-Lab)

**学习 / 研究实践 · Fork**

整理 GPU、AI Systems、推理服务和性能工程相关实验。该仓库用于学习和资料沉淀，非原创上游项目。

### [Single-Card-Multi-Model-Colocation](https://github.com/world-cyj/Single-Card-Multi-Model-Colocation)

面向单卡多模型共置、资源分配与推理调度的实验项目，记录多模型服务场景下的资源利用问题。

### [npu_ie](https://github.com/world-cyj/npu_ie)

NPU 相关工程实践仓库，记录设备侧推理、模型执行和异构算力方向的实验内容。

## 公开仓库地图

### AI Systems / Performance

- [MinivLLM_plus](https://github.com/world-cyj/MinivLLM_plus) · Mini-vLLM 推理引擎学习与实现
- [AI-System-Performance-Lab](https://github.com/world-cyj/AI-System-Performance-Lab) · 学习 / 研究实践，Fork
- [Single-Card-Multi-Model-Colocation](https://github.com/world-cyj/Single-Card-Multi-Model-Colocation) · 单卡多模型共置
- [npu_ie](https://github.com/world-cyj/npu_ie) · NPU 工程实践
- [vllm-omni](https://github.com/world-cyj/vllm-omni) · vLLM Omni 学习，Fork

### Backend / Infra

- [adapnpu_serve](https://github.com/world-cyj/adapnpu_serve) · 推理服务与调度实验
- [12306_demo](https://github.com/world-cyj/12306_demo) · Java 后端项目实践
- [MediFlow](https://github.com/world-cyj/MediFlow) · JavaScript 应用项目
- [LLM_codespace](https://github.com/world-cyj/LLM_codespace) · LLM 学习空间
- [agent-skills](https://github.com/world-cyj/agent-skills) · AI Coding Agent Skills，Fork

### Learning Archive

- [how-to-optim-algorithm-in-cuda](https://github.com/world-cyj/how-to-optim-algorithm-in-cuda) · CUDA 算法优化学习，Fork
- [dive_into_deep_learning](https://github.com/world-cyj/dive_into_deep_learning) · 深度学习课程笔记
- [LeetCode-Go](https://github.com/world-cyj/LeetCode-Go) · Go 语言算法题解，Fork
- [AutoAgent](https://github.com/world-cyj/AutoAgent) · SpringAI Agent 学习，Fork
- [claude-code-deep-dive](https://github.com/world-cyj/claude-code-deep-dive) · Claude Code 源码研究报告

## 技术栈

**编程语言**  
`Go` `C++` `Python` `Java`

**后端与平台**  
`REST API` `gRPC` `Beego` `MySQL` `Redis` `Docker` `Kubernetes` `client-go`

**AI Infra**  
`vLLM` `SGLang` `TensorRT-LLM` `Triton` `PyTorch` `PagedAttention` `KV Cache` `Continuous Batching`

**计算与通信**  
`CUDA` `CUTLASS` `CUDA Graph` `GEMM` `NCCL` `HCCL` `RDMA` `RoCE` `D2D`

**观测与排障**  
`Prometheus` `Grafana` `Logs` `Trace` `GDB` `CMake` `nsys` `ncu`

## 研究与输出

- 第一作者专利交底：一种分域 AI 加速器多租户推理的资源形态感知调度方法及系统。
- 第一作者论文投稿：`A Review of Optimization Techniques for Large Language Model Inference`，已投稿至 KSEM 2025，状态以投稿记录为准。
- 第一作者论文投稿：`AdapDomain: Residual-Shape Scheduling for Multi-Tenant Inference on Domain-Partitioned Accelerators`，已投稿至 SoCC 2026，状态以投稿记录为准。
- 研究生学业奖学金一等奖，英语六级（CET-6）。

## 教育背景

- **深圳大学** · 计算机科学与技术硕士 · `2024.09 – 2027.06`
- **天津科技大学** · 自动化本科 · `2020.09 – 2024.06`

## 联系

深圳，中国  
[cyj2582329754@gmail.com](mailto:cyj2582329754@gmail.com)

<div align="center">

`Systems in motion.`

</div>
