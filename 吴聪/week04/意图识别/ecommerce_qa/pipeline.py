# -*- coding: utf-8 -*-
"""在线问答主流程（RAG Pipeline）。

流程：
  用户问题 -> 意图识别 -> 实体抽取 -> 对话管理（槽位检查）
         -> 知识检索（图检索 + BM25）-> 生成（Deepseek / 模板）
"""
from ecommerce_qa import config
from ecommerce_qa.generator import templates
from ecommerce_qa.generator.deepseek_llm import generate
from ecommerce_qa.knowledge.memory_store import MemoryGraphStore
from ecommerce_qa.ner.predict import NerPredictor
from ecommerce_qa.nlu.dialog import DialogManager
from ecommerce_qa.nlu.entity_extractor import EntityExtractor
from ecommerce_qa.nlu.intent_model import IntentClassifier
from ecommerce_qa.relation_extraction.predict import RelPredictor
from ecommerce_qa.retrieval.bm25 import BM25Index
from ecommerce_qa.retrieval.graph_retrieval import GraphRetriever
from ecommerce_qa.utils.logger import get_logger

logger = get_logger(__name__)


def _rule_intent(text: str) -> str:
    """基于关键词的意图兜底（意图模型缺失时的备选）。"""
    if any(k in text for k in ["退货", "换货", "退款", "售后", "坏了", "开胶", "瑕疵"]):
        return "咨询售后"
    if any(k in text for k in ["订单", "物流", "发货", "签收", "快递", "包裹"]):
        return "查订单状态"
    if any(k in text for k in ["无理由", "退换货", "运费", "拒收", "政策"]):
        return "查退换货政策"
    if any(k in text for k in ["你好", "谢谢", "你是谁", "再见", "在吗", "哈哈"]):
        return "闲聊"
    return "查商品参数"


class EcommerceQA:
    """电商智能问答系统主入口。"""

    def __init__(self):
        self.ner = NerPredictor.load()
        self.rel = RelPredictor.load()
        self.intent = IntentClassifier.load()
        self.extractor = EntityExtractor(
            ner_predictor=self.ner,
            known_entities=EntityExtractor.build_known_entities(),
        )
        self.dialog = DialogManager()
        self.store = None
        self.bm25 = None
        self._load_graph()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _load_graph(self) -> None:
        """加载已构建的知识图谱与 BM25 索引。"""
        graph_path = config.GRAPH_DIR / "graph.json"
        if graph_path.exists():
            self.store = MemoryGraphStore.load_json(graph_path)
            logger.info("已加载知识图谱：%d 节点 / %d 关系",
                        self.store.node_count, self.store.edge_count)
        else:
            logger.warning("未找到知识图谱，请先运行 scripts/04_build_graph.py")

        bm25_path = config.GRAPH_DIR / "bm25_index.json"
        if bm25_path.exists():
            self.bm25 = BM25Index.load_json(bm25_path)
            logger.info("已加载 BM25 索引：%d 篇文档", self.bm25.N)

    def status(self) -> dict:
        """返回各模块的就绪状态（供界面展示）。"""
        return {
            "intent模型": self.intent is not None,
            "NER模型": self.ner is not None,
            "关系模型": self.rel is not None,
            "知识图谱": self.store is not None,
            "BM25索引": self.bm25 is not None,
            "Deepseek": config.has_deepseek_key(),
        }

    # ------------------------------------------------------------------
    # 知识检索
    # ------------------------------------------------------------------
    def _retrieve(self, question: str, slots: dict) -> list:
        """合并图检索 + BM25 检索结果，返回知识字符串列表（去重）。"""
        knowledge = []

        # 1) 图检索：沿槽位实体查邻居三元组
        if self.store is not None:
            entities = [v for values in slots.values() for v in values]
            if entities:
                retriever = GraphRetriever(self.store)
                for s, r, o in retriever.search(entities, top_k=10):
                    knowledge.append(f"{s} {r} {o}")

        # 2) BM25 检索：在商品/政策文档上做文本相关性匹配
        if self.bm25 is not None:
            for idx, _score in self.bm25.search(question, top_k=3):
                knowledge.append(self.bm25.docs[idx])

        return list(dict.fromkeys(knowledge))

    # ------------------------------------------------------------------
    # 主回答流程
    # ------------------------------------------------------------------
    def answer(self, question: str, session: dict) -> dict:
        """回答一个问题，返回完整的结果 dict。

        Args:
            question: 用户问题
            session: 会话状态（含 "slots"），多轮对话时传同一个 dict

        Returns:
            {"question", "intent", "intent_prob", "slots", "strategy",
             "knowledge", "answer", "source"}
        """
        # 1) 意图识别
        if self.intent is not None:
            intent, prob = self.intent.predict(question)
        else:
            intent, prob = _rule_intent(question), 0.0
            logger.warning("意图模型缺失，使用规则兜底")

        # 2) 实体抽取
        slots = self.extractor.extract(question)

        # 3) 对话管理：槽位填充 + 缺失检查
        state = self.dialog.update(session, intent, slots)

        # 4) 缺槽位 -> 反问
        if state["missing"]:
            missing = state["missing"][0]
            return {
                "question": question,
                "intent": intent,
                "intent_prob": prob,
                "slots": state["slots"],
                "strategy": "ask_missing",
                "knowledge": [],
                "answer": self.dialog.ask_message(missing),
                "source": "ask",
            }

        # 5) 闲聊
        if intent == "闲聊":
            return {
                "question": question,
                "intent": intent,
                "intent_prob": prob,
                "slots": state["slots"],
                "strategy": "chat",
                "knowledge": [],
                "answer": templates.chat_reply(question),
                "source": "chat",
            }

        # 6) 知识检索
        knowledge = self._retrieve(question, state["slots"])

        # 7) 生成回答
        if knowledge:
            answer, source = generate(
                question, knowledge, intent=intent, slots=state["slots"]
            )
            strategy = "generate"
        else:
            answer = templates.no_knowledge_reply(intent)
            source = "none"
            strategy = "no_knowledge"

        return {
            "question": question,
            "intent": intent,
            "intent_prob": prob,
            "slots": state["slots"],
            "strategy": strategy,
            "knowledge": knowledge,
            "answer": answer,
            "source": source,
        }
