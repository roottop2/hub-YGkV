# -*- coding: utf-8 -*-
"""电商智能问答系统 核心代码包。

分层：
    data_generation  模拟数据生成与自动标注
    ner              BiLSTM+CRF 命名实体识别
    relation_extraction  R-BERT 关系抽取
    knowledge        知识融合与图谱存储
    retrieval        BM25 检索与图检索
    nlu              BERT 意图识别 / 实体抽取 / 对话管理
    generator        Deepseek 生成器与降级模板
    pipeline         串联在线问答主流程
"""
