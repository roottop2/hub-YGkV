# 电商智能问答系统

基于**知识图谱 + RAG（检索增强生成）**的电商智能客服问答系统。
用户用自然语言提问，系统先**理解意图、抽取实体**，再从**知识图谱 + BM25 索引**中检索真实知识，最后交给 **Deepseek 大模型**生成准确回答，有效压制"幻觉"。

> 技术路线与设计文档见 `docs/技术路线.md`，适用场景见 `docs/技术路线适用场景.md`。

---

## ✨ 功能

- **多轮对话**：订单查询缺订单号时会反问补全
- **知识图谱检索**：品牌/型号/订单/政策等结构化知识（366 节点 / 665 关系）
- **BM25 全文检索**：商品参数、售后政策文本召回
- **Deepseek 生成**：基于检索到的知识生成自然回答，无 Key 自动降级模板回答
- **可视化界面**：Streamlit 网页聊天，可展开查看"意图/槽位/检索知识"调试信息

## 🧠 技术栈

| 环节 | 技术 |
|---|---|
| 命名实体识别 NER | BiLSTM + 自实现 CRF |
| 关系抽取 | R-BERT（带实体标记的 BERT） |
| 意图识别 | 微调 BERT（bert-base-chinese） |
| 知识融合 | TF-IDF 实体消歧 + 规则统一 + 同义词对齐 |
| 知识图谱存储 | networkx 内存图（抽象接口，可切 Neo4j） |
| 检索 | 自实现 BM25 + 图检索 |
| 生成 | Deepseek API（OpenAI 兼容）+ 降级模板 |
| 界面 | Streamlit |

## 📁 目录结构

```
意图识别/
├── README.md                  # 本文档
├── requirements.txt           # 依赖
├── .env.example               # Deepseek API Key 配置模板
├── docs/                      # 技术路线 / 适用场景 / 实现计划
├── data/
│   ├── raw/                   # 模拟原始数据（自动生成）
│   ├── annotated/             # 自动标注的训练数据
│   └── graph/                 # 知识图谱 + BM25 索引
├── ecommerce_qa/              # 核心代码包
│   ├── data_generation/       # 模拟数据生成 + 自动标注
│   ├── ner/                   # BiLSTM+CRF 命名实体识别
│   ├── relation_extraction/   # R-BERT 关系抽取
│   ├── knowledge/             # 知识融合 + 图谱存储
│   ├── retrieval/             # BM25 + 图检索
│   ├── nlu/                   # 意图识别 + 实体抽取 + 对话管理
│   ├── generator/             # Deepseek 生成器
│   └── pipeline.py            # 问答主流程
├── scripts/                   # 按编号执行
│   ├── 01_generate_data.py    # 生成数据与标注
│   ├── 02_train_ner.py        # 训练 NER
│   ├── 03_train_relation.py   # 训练关系抽取
│   ├── 04_build_graph.py      # 构建知识图谱 + BM25
│   ├── 05_train_intent.py     # 训练意图识别
│   ├── 06_evaluate.py         # 评测
│   ├── 07_demo_cli.py         # 命令行演示
│   └── run_all.py             # 一键全流程
├── app/streamlit_app.py       # Streamlit 界面
└── models/                    # 训练产物
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 使用 conda 的 python_ai 环境（Windows Git Bash）
conda activate python_ai

# 安装依赖（如缺失）
pip install -r requirements.txt
```

### 2. 配置 Deepseek API Key（可选）

复制 `.env.example` 为 `.env`，填入你的 Key：

```
DEEPSEEK_API_KEY=你的key
```

> 不配置也能运行：系统会自动降级为模板回答。
> 申请地址：https://platform.deepseek.com

### 3. 一键跑通全流程

```bash
python scripts/run_all.py
```

这一步依次：生成模拟数据 → 训练 NER → 训练关系抽取 → 构建知识图谱 → 训练意图识别 → 评测。

### 4. 启动界面

```bash
streamlit run app/streamlit_app.py
```

或先试用命令行：

```bash
python scripts/07_demo_cli.py
```

## 🧪 评测

`python scripts/06_evaluate.py` 输出（快速模式示例）：

| 指标 | 数值 |
|---|---|
| NER F1 | 0.87 |
| 关系抽取准确率 | 1.00 |
| 意图分类准确率 | 1.00 |
| 端到端满意度模拟 | 0.85 |

> 以上为少量模拟数据 + CPU 快速训练的结果，用于验证**流程正确性**。
> 生产指标（F1 91.26%、节点 31680、满意度 93%）的复现方法见设计文档《生产化方案》。

## ⚙️ 工作原理（RAG 流程）

```
用户问题
  │
  ├─ 意图识别（微调 BERT）→ 查订单/查商品/售后/政策/闲聊
  ├─ 实体抽取（词典 + NER + 正则）→ 订单号/品牌/型号…
  ├─ 对话管理（槽位检查）→ 缺订单号？反问！
  │
  ├─ 知识检索：图检索（实体→三元组） + BM25（文本相关度）
  │
  └─ 生成：Deepseek(问题 + 检索到的知识) → 回答（失败降级模板）
```

## ❓ 常见问题

**Q：BERT 预训练模型下载失败？**
网络受限时可设置镜像后重试：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Q：训练很慢？**
本机无 GPU，代码默认"快速模式"（小数据、少 epoch，CPU 约 1 分钟跑完）。
`ecommerce_qa/config.py` 中 `QUICK_MODE` 与各模型超参可调。

**Q：Deepseek 回答出现"来源: template"？**
说明未配置 Key 或调用失败，系统已降级为模板回答。配置 `.env` 后重启即可。

**Q：怎么扩展成真实业务数据？**
替换 `data/raw/` 为真实数据，调整 `config.py` 的实体/关系/意图定义即可（详见设计文档）。
