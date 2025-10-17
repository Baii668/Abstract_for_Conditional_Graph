# -*- coding: utf-8 -*-
from openai import OpenAI
from PROMPT import *
from typing import List, Dict, Any, Optional, Literal, Tuple
import numpy as np
import math
from FlagEmbedding import BGEM3FlagModel, FlagReranker
import re
import os
import json

client = OpenAI(api_key="", base_url="https://api.deepseek.com")
def qa_openai(prompt: str) -> str:

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=0
    )
    return response.choices[0].message.content


def create_multi_aspect_query(strategy: str) -> List[str]:
    base = (strategy or "").strip()
    json_format = """相关情况1;相关情况2;....."""
    prompt = MULTI_ASPECT_QUERY.format(
        base=base,
        json_format=json_format
    )
    while True:
        try:
            raw_query = qa_openai(prompt)
            if ";" in raw_query:
                break
        except TypeError:
            print("Error")

    return raw_query

def to_list(raw_query: str) -> List[str]:
    """
    将以分号（; 或 ；）分隔的内容拆分为 List[str]。
    例如："处置策略A; 处置策略B；处置策略C;;" -> ["处置策略A","处置策略B","处置策略C"]
    """
    if not raw_query:
        return []

    # 按中英文分号切分，连续分号视作一个分隔符
    segments = re.split(r"[;；]+", raw_query.strip())

    # 清洗空白并过滤空项
    result = [seg.strip() for seg in segments if seg and seg.strip()]
    return result

def generate_keywords(sentence):
    json_format = """相关情况1;相关情况2;....."""
    prompt = GENERATE_KEYWORDS.format(
        base=sentence,
        json_format=json_format
    )
    while True:
        try:
            raw_sentence = qa_openai(prompt)
            if ";" in raw_sentence:
                break
        except TypeError:
            print("Error")
    key_words = to_list(raw_sentence)
    return key_words


MatchMode = Literal["any", "all"]
Agg = Literal["max", "mean", "sum"]  # 多查询分数聚合

class BGERecall:
    """
    bge-m3 建索引；输入 strategy 文本 -> 生成多查询；
    先只召回候选（不排序），再用 bge-reranker-v2-m3 精排。
    """

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _sparse_dot(q: Dict[int, float], d: Dict[int, float]) -> float:
        if not q or not d:
            return 0.0
        if len(q) > len(d):
            q, d = d, q
        s = 0.0
        for k, v in q.items():
            if k in d:
                s += v * d[k]
        return float(s)

    def __init__(
        self,
        model_dir: str = "BAAI/bge-m3",
        reranker_dir: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
    ):
        self.model = BGEM3FlagModel(model_dir, use_fp16=use_fp16)
        self.reranker = FlagReranker(reranker_dir, use_fp16=use_fp16)
        self.texts: List[str] = []
        self._dense: List[np.ndarray] = []
        self._sparse: List[Dict[int, float]] = []
        self.metas: List[Dict[str, Any]] = []

    # 1) 建索引
    def build_index(self, texts: List[str], metas: Optional[List[Dict[str, Any]]] = None):
        self.texts = [(t or "").strip() for t in texts]
        enc = self.model.encode_corpus(
            self.texts, return_dense=True, return_sparse=True, return_colbert_vecs=False
        )
        self._dense  = [np.asarray(v, dtype=np.float32) for v in enc["dense_vecs"]]
        self._sparse = enc["lexical_weights"]
        self.metas   = [{} for _ in self.texts] if metas is None else metas

    # 2) 基于 strategy 生成多查询
    def gen_queries(self, strategy: str, *, dedup: bool = True, multi_query: bool = True) -> List[str]:
        if multi_query:
            raw = create_multi_aspect_query(strategy)  # -> "子查询1;子查询2;..."
            queries = to_list(raw)  # -> List[str]
        else:
            queries = [(strategy or "").strip()] if strategy else []

        # 去重、裁剪
        if dedup:
            seen, uniq = set(), []
            for q in queries:
                if q and q not in seen:
                    seen.add(q)
                    uniq.append(q)
            queries = uniq

        # 兜底：如果为空
        if not queries:
            qs = (strategy or "").strip()
            queries = [qs] if qs else []
        return queries

    # 3) 只召回候选（不排序）
    def recall_candidates(
        self,
        queries: List[str],
        *,
        use_dense: bool = True,
        use_sparse: bool = True,
        # 方式A：阈值过滤
        dense_threshold: float = 0.35,
        sparse_threshold: float = 1.0,
        match_mode: MatchMode = "any",
        # 方式B：每查询TopN（>0 则启用）
        topn_per_query_dense: int = 0,
        topn_per_query_sparse: int = 0,
    ) -> List[int]:
        """
        返回候选文档的 index（按原始顺序去重保留）。只召回，不排序。
        """
        assert len(self.texts) == len(self._dense) == len(self._sparse), "Index not built or inconsistent."
        qs = [q.strip() for q in queries if q and q.strip()]
        if not qs:
            return []

        # 编码查询
        q_dense_vecs, q_sparse_vecs = [], []
        if use_dense:
            enc_qd = self.model.encode_queries(qs, return_dense=True, return_sparse=False, return_colbert_vecs=False)
            q_dense_vecs = [np.asarray(v, dtype=np.float32) for v in enc_qd["dense_vecs"]]
        if use_sparse:
            enc_qs = self.model.encode_queries(qs, return_dense=False, return_sparse=True, return_colbert_vecs=False)
            q_sparse_vecs = enc_qs["lexical_weights"]

        n = len(self.texts)

        # ---- 模式B：每查询 TopN ----
        if (topn_per_query_dense > 0) or (topn_per_query_sparse > 0):
            cand: List[int] = []
            if use_dense and topn_per_query_dense > 0:
                for qv in q_dense_vecs:
                    scores = [self._cosine(qv, self._dense[i]) for i in range(n)]
                    order = sorted(range(n), key=lambda i: scores[i], reverse=True)[:topn_per_query_dense]
                    cand.extend(order)
            if use_sparse and topn_per_query_sparse > 0:
                for qv in q_sparse_vecs:
                    scores = [self._sparse_dot(qv, self._sparse[i]) for i in range(n)]
                    order = sorted(range(n), key=lambda i: scores[i], reverse=True)[:topn_per_query_sparse]
                    cand.extend(order)
            # 原始顺序去重
            seen = set(); out = []
            for i in range(n):
                if i in cand and i not in seen:
                    seen.add(i); out.append(i)
            return out

        # ---- 模式A：阈值过滤（不排序）----
        out_idx: List[int] = []
        for i in range(n):
            dense_scores = [self._cosine(qv, self._dense[i]) for qv in q_dense_vecs] if use_dense else []
            sparse_scores = [self._sparse_dot(qv, self._sparse[i]) for qv in q_sparse_vecs] if use_sparse else []
            per_query_hit = []
            for qi in range(len(qs)):
                hit_dense  = (dense_scores[qi]  >= dense_threshold)  if use_dense  else False
                hit_sparse = (sparse_scores[qi] >= sparse_threshold) if use_sparse else False
                per_query_hit.append(hit_dense or hit_sparse)
            matched = any(per_query_hit) if match_mode == "any" else all(per_query_hit)
            if matched:
                out_idx.append(i)
        return out_idx

    def rank_with_reranker(
             self,
            queries: List[str],
            candidate_indices: List[int],
            *,
            agg: Agg = "max",
            top_k: int = 10,
            batch_size: int = 64,
            use_keywords: bool = True,
            kw_weight: float = 1.5,
            max_keywords: int = 4,
            event_type: Optional[str] = None,
            strategy: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not candidate_indices:
            return []

        # 如果启用关键词
        kw_queries: List[str] = []
        if use_keywords and strategy:
            print("if use_keywords and strategy")
            kw_queries = generate_keywords(strategy)
            if event_type:
                if isinstance(event_type, list):
                    kw_queries.extend(event_type)
                elif isinstance(event_type, str):
                    kw_queries.append(event_type)

        print("关键词 queries:", kw_queries)

        # 组装 (q, doc) 对
        pairs: List[Tuple[str, str]] = []
        owner: List[int] = []
        query_origin: List[str] = []  # 记录 query 是原始还是关键词
        for idx in candidate_indices:
            d = self.texts[idx]
            for q in queries:
                pairs.append((q, d))
                owner.append(idx)
                query_origin.append("base")
            for q in kw_queries:
                pairs.append((q, d))
                owner.append(idx)
                query_origin.append("kw")

        # 打分
        scores = self.reranker.compute_score(pairs, batch_size=batch_size)

        # 聚合
        per_doc: Dict[int, List[Tuple[str, float, str]]] = {i: [] for i in candidate_indices}
        for (q, _), s, idx, origin in zip(pairs, scores, owner, query_origin):
            per_doc[idx].append((q, float(s), origin))

        def agg_fn(vals: List[Tuple[str, float, str]]) -> float:
            if not vals: return 0.0
            base_scores = [s for _, s, o in vals if o == "base"]
            kw_scores   = [s for _, s, o in vals if o == "kw"]

            if agg == "max":
                base = max(base_scores) if base_scores else 0.0
                kw   = max(kw_scores)   if kw_scores   else 0.0
            elif agg == "mean":
                base = sum(base_scores)/len(base_scores) if base_scores else 0.0
                kw   = sum(kw_scores)/len(kw_scores)     if kw_scores   else 0.0
            else:  # sum
                base = sum(base_scores)
                kw   = sum(kw_scores)

            return base + kw_weight * kw

        ranked = []
        for i in candidate_indices:
            q_s = per_doc[i]
            final = agg_fn(q_s)
            ranked.append({
                "doc_index": i,
                "rank_score": float(final),
                "text": self.texts[i],
                "meta": self.metas[i],
                "per_query_scores": q_s
            })

        ranked.sort(key=lambda x: x["rank_score"], reverse=True)
        for r, it in enumerate(ranked, 1):
            it["rank"] = r
        return ranked[:top_k]

    # 5) 一步到位：strategy -> 多查询 -> 候选召回 -> 精排
    def search_ranked(
            self,
            strategy: str,
            *,
            # 召回
            topn_per_query_dense: int = 20,
            topn_per_query_sparse: int = 10,
            use_dense: bool = True,
            use_sparse: bool = True,
            dense_threshold: float = 0.0,
            sparse_threshold: float = 0.0,
            match_mode: MatchMode = "any",
            # 精排
            agg: Agg = "max",
            top_k: int = 10,
            batch_size: int = 64,
            event_type=None,
            # 新增参数
            multi_query: bool = True,
            use_keywords: bool = True
    ) -> Dict[str, Any]:
        queries = self.gen_queries(strategy, multi_query=multi_query)
        cand_idx = self.recall_candidates(
            queries,
            use_dense=use_dense,
            use_sparse=use_sparse,
            dense_threshold=dense_threshold,
            sparse_threshold=sparse_threshold,
            match_mode=match_mode,
            topn_per_query_dense=topn_per_query_dense,
            topn_per_query_sparse=topn_per_query_sparse,
        )
        ranked = self.rank_with_reranker(
            queries, cand_idx, agg=agg, top_k=top_k, batch_size=batch_size, event_type=event_type, use_keywords=use_keywords,
            strategy=strategy
        )
        return {"queries": queries, "candidates": cand_idx, "ranked": ranked}


def find_summary_list(folder_path, model_dir_bge_m3, model_dir_bge_reranker_v2_m3, local_summary, event_type,
                      topn_per_query_dense: int = 20,
                      topn_per_query_sparse: int = 10,
                      agg: str = "max",
                      top_k: int = 10
                      ):
    summary_list = []
    back_check_dict = {}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            tag_str = data["label_org"][0] + " " + data["label_org"][1] + " " + data["label_org"][2] + " " + data["label_actor"][0] + " " + data["label_actor"][1] + " " + data["label_actor"][2] + " "+ data["label_type"][0] + " " + data["label_type"][1] + " " + data["label_type"][2]
            # tag_str = data["event_name"]
            summary_list.append(tag_str)
            back_check_dict[tag_str] = filename

    engine = BGERecall(
        model_dir=model_dir_bge_m3,
        reranker_dir=model_dir_bge_reranker_v2_m3,
        use_fp16=True
    )
    engine.build_index(summary_list)

    result = engine.search_ranked(
        strategy=local_summary,
        topn_per_query_dense=topn_per_query_dense,
        topn_per_query_sparse=topn_per_query_sparse,
        agg=agg,
        top_k=top_k,
        event_type=event_type,
        multi_query=False,  # 只用 strategy 本身，不生成多查询
        use_keywords=False
    )

    show_passage_list = []
    show_summary_list = []
    for summary in result["ranked"]:
        show_passage_list.append(back_check_dict[summary['text']])
        show_summary_list.append(summary['text'])

    return show_passage_list, show_summary_list


def find_outcome_list(folder_path, model_dir_bge_m3, model_dir_bge_reranker_v2_m3, local_outcome, event_type,
                      topn_per_query_dense: int = 20,
                      topn_per_query_sparse: int = 10,
                      agg: str = "max",
                      top_k: int = 10
                    ):
    outcome_list = []
    back_check_dict = {}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for o in data["outcome_new"]:
                outcome_list.append(o["outcome_tag"] + "/" + o["summary"])
                back_check_dict[o["outcome_tag"] + "/" + o["summary"]] = filename

    engine = BGERecall(
        model_dir=model_dir_bge_m3,
        reranker_dir=model_dir_bge_reranker_v2_m3,
        use_fp16=True
    )
    engine.build_index(outcome_list)

    result = engine.search_ranked(
        strategy=local_outcome,
        topn_per_query_dense=topn_per_query_dense,
        topn_per_query_sparse=topn_per_query_sparse,
        agg=agg,
        top_k=top_k,
        event_type=event_type,
        multi_query=True
    )

    show_passage_list = []
    show_outcome_list = []
    for outcome in result["ranked"]:
        show_passage_list.append(back_check_dict[outcome['text']])
        show_outcome_list.append(outcome['text'])

    return show_passage_list, show_outcome_list


def find_matching_edges(data, outcome_str):
    # 提取 / 前面的内容
    prefix = outcome_str.split('/')[0]
    print(prefix)
    for edge in data["new_edges"]:
        print(edge["new_end"]["outcome_tag"])
        if edge["new_end"]["outcome_tag"].startswith(prefix):
            return {
                "new_start": edge["new_start"],
                "new_end": edge["new_end"]
            }
    return None


if __name__ == "__main__":
    folder_path = "/home/jianbaizhao/CodingFile/IDM/pipeline/out_data_company"


    model_dir_bge_m3 = "/home/jianbaizhao/model/BAAI/bge-m3"
    model_dir_bge_reranker_v2_m3 ="/home/jianbaizhao/model/BAAI/bge-reranker-v2-m3"

    local_summary = "灾难——生产事故——责任单位"
    event_type  =["企业", "政府", "生产事故", "爆炸事故", "安全生产监管机构"]
    local_outcome = "事故挂牌督办"
    outcome_type = "国务院安委会根据相关法律对该起事故实施了挂牌督办。"

    summary_passage_list, summary_list = find_summary_list(
        folder_path=folder_path,
        model_dir_bge_m3=model_dir_bge_m3,
        model_dir_bge_reranker_v2_m3=model_dir_bge_reranker_v2_m3,
        local_summary=local_summary,
        event_type=event_type,
        topn_per_query_dense=60,
        topn_per_query_sparse=30,
        agg="max",
        top_k=30
    )
    all_similar_event_strategy = []
    for run in range(3):
        print(f"第 {run + 1} 次运行:")
        outcome_passage_list,  outcome_list= find_outcome_list(
            folder_path=folder_path,
            model_dir_bge_m3=model_dir_bge_m3,
            model_dir_bge_reranker_v2_m3=model_dir_bge_reranker_v2_m3,
            local_outcome=local_outcome,
            event_type=outcome_type,
            topn_per_query_dense=60,
            topn_per_query_sparse=30,
            agg="max",
            top_k=30
        )
        print("Summary:-----#####-----#####-----")
        for s in summary_passage_list:
            print(s)
        print("outcome:-----#####-----#####-----")
        for o in outcome_passage_list:
            print(o)
        print("Summary&&&outcome:-----#####-----#####-----")
        intersection = list(set(summary_passage_list) & set(outcome_passage_list))
        print(intersection)

        similar_event_strategy = []
        for filename in intersection:
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            outcome_index = outcome_passage_list.index(filename)
            corresponding_outcome = outcome_list[outcome_index]

            matching_edges = find_matching_edges(data, corresponding_outcome)
            print(str(filename))
            a_dict = {
                str(filename):{
                    "new_start": matching_edges["new_start"],
                    "new_end": matching_edges["new_end"],
                    "new_basis": data["new_basis"]
                }
            }
            similar_event_strategy.append(a_dict)
        all_similar_event_strategy.extend(similar_event_strategy)
        print(f"第 {run + 1} 次运行完成，找到 {len(similar_event_strategy)} 个策略\n")

    # 去重：基于文件名去重
    unique_strategies = {}
    for strategy in all_similar_event_strategy:
        for filename, content in strategy.items():
            if filename not in unique_strategies:
                unique_strategies[filename] = content
    # 转换回列表格式
    final_similar_event_strategy = [{filename: content} for filename, content in unique_strategies.items()]
    with open("similar_event_strategy.json", "w", encoding="utf-8") as f:
        json.dump(final_similar_event_strategy, f, ensure_ascii=False, indent=4)
    print(f"总共找到 {len(final_similar_event_strategy)} 个唯一策略，已保存到 similar_event_strategy.json")




