# -*- coding: utf-8 -*-

import openai
from openai import OpenAI
from typing import List, Dict, Optional, Any
import json
import os
import re
from collections import defaultdict, Counter
import pandas as pd
import ast
from PROMPT import *
from Atomic_decompose import *
# OPENAI_API_KEY = "\"
# OPENAI_BASE_URL = "https://api.gpt.ge/v1/chat/completions"
# OPENAI_MODEL_NAME = "gpt-4-turbo-2024-04-09"
# client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
client = OpenAI(api_key="", base_url="https://api.deepseek.com")
def qa_openai(prompt: str, tem = 1) -> str:

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=tem
    )
    return response.choices[0].message.content

similar_events = [
    "155_如何看待华中农业大学黄某若教授被课题组十一名成员联合举报学术造假？.json",
    "114_云南高校副院长学位论文抄袭.json",
    "386_西安电子科大通报学生毕设代做事件.json",
    "626_北邮通报学生联名举报导师事件.json",
    "670_中央美院教授落马后论文被认定抄袭.json"
]

def counterfact_find(classified_basis_list):
    '''
    todo
    :param file:
    :return:
    '''
    result = {
        "common": "",
        "seldom": ""
    }
    return result

def create_common_tag(common_basis:dict):
    for key, value in common_basis.items():
        basis_summary = key
        basis_items = value
    str_basis_items = ""
    n = 1
    for i in basis_items:
        str_basis_items += "item" + str(n) + ":" + i + "\n"
        n += 1
    base_prompt = COMMON_SUMMARY.format(
        basis_summary=basis_summary,
        basis_items=str_basis_items
    )
    feedback = None
    while True:
        try:
            if feedback:
                prompt1 = f"{base_prompt}\n注意：请结合专家反馈意见『{feedback}』重新总结概括。"
            else:
                prompt1 = base_prompt
            raw_summary = qa_openai(prompt1)

            prompt2 = CHECK_SUMMARY.format(
                basis_summary=basis_summary,
                basis_items=str_basis_items,
                raw_summary=raw_summary
            )
            check_summary = qa_openai(prompt2)
            if check_summary == "同意":
                break
            else:
                match = re.search(r"不同意(.*)", check_summary)
                if match:
                    feedback = match.group(1).strip()
                    print("extracted feedback:", feedback)
                    continue  # 回到 while 顶部，重新生成 prompt1（带反馈版）
                else:
                    print("Warning: check_summary 格式不符合预期:", check_summary)
                    continue
        except Exception as e:
            print("ERROR:", e)
            continue

    return raw_summary


def create_common_fact(common_basis_tag_list):
    i = 1
    all_common_des = ""
    for c in common_basis_tag_list:
        for key, value in c.items():
            all_common_des += str(i) + ":\t" + key + "是" + value + "\n"
            i += 1
    json_format = """
‘’‘
    [事件背景]:XXX
    [起因]:XXX
    [经过]:XXX
    [影响]:XXX
    [完整的新闻报道]:XXX
’‘’
"""
    prompt = COMMON_FACT.format(
        common_des=all_common_des,
        json_format=json_format
    )
    # print(prompt)
    while True:
        try:
            result = qa_openai(prompt)
            print(result)
            pattern = r'\[完整的新闻报道\]\s*[:：]?\s*["“”]?(.*?)["“”]?(?:,|\}|$)'
            match = re.search(pattern, result, re.S)
            if match:
                common_fact = match.group(1).strip()
                print("extracted common_fact:", common_fact)
                break
            else:
                print("Warning: common fact:", result)
        except Exception as e:
            print("ERROR:", e)

    return common_fact

def create_counter_seldom_basis(seldom_basis: dict):
    for key, value in seldom_basis.items():
        basis_summary = key
        basis_items = value
    """
        对少数条件取反
    """
    prompt1 = COUNTER_SELDOM_SUMMARY.format(
        basis_summary=basis_summary,
        basis_items=basis_items
    )
    while True:
        try:
            counter_seldom_summary = qa_openai(prompt=prompt1,tem=0.5)
            match = re.search(r"[\]:：]\s*(.*)", counter_seldom_summary)
            if match:
                content = match.group(1).strip()
                items = re.split(r"[;；]", content)
                counter_seldom_summary_list = [i.strip() for i in items if i.strip()]
                if len(counter_seldom_summary_list) != 0:
                    break
                else:
                    print("Warning: list为空:", counter_seldom_summary)
            else:
                print("Warning: 格式不符合预期:", counter_seldom_summary)
                continue
        except Exception as e:
            print("ERROR:", e)

    return counter_seldom_summary_list

def common_add_seldom_fact(common_fact: str, seldom_basis: dict, counter_seldom_basis: str):
    for key, value in seldom_basis.items():
        basis_summary = key
    a_seldom_des = basis_summary + ":" + counter_seldom_basis
    print("seldom_des:", a_seldom_des)
    """
        与一般条件拼接
    """
    prompt2 = COMMON_ADD_SELDOM_FACT.format(
        common_fact=common_fact,
        seldom_tag=a_seldom_des
    )
    common_seldom_fact = qa_openai(prompt2)

    return common_seldom_fact


def check_seldom_basis(fact:str, strategy:str, basis_summary, counter_basis_items):

    important_basis = basis_summary + "是" + counter_basis_items
    prompt = CHECK_BASIS.format(
        fact=fact,
        strategy=strategy,
        seldom_basis=important_basis
    )
    while True:
        try:
            check_basis = qa_openai(prompt=prompt, tem=1.5)
            if check_basis == "同意" or check_basis == "不同意":
                break
            else:
                print("Warning:", check_basis)
        except Exception as e:
            print("ERROR:", e)
    return check_basis

def find_basis(seldom_b, seldom_basis_tag_list):
    print("seldom_b:", seldom_b)
    print("seldom_basis_tag_list:", seldom_basis_tag_list)
    for key, value in seldom_b.items():
        this_key = key
    for s in seldom_basis_tag_list:
        for a_key, a_value in s.items():
            if this_key == a_key:
                return  s

    raise Exception(f"没有找到匹配的key: {this_key}")

def create_basis_tag_list(common_basis: list):
    common_basis_tag_list = []
    for common_b in common_basis:
        print(common_b)
        raw_a_common_b_tag = create_common_tag(common_b)
        print(raw_a_common_b_tag)
        for key, value in common_b.items():
            common_basis_tag_list.append({
                key: raw_a_common_b_tag
            })
    return common_basis_tag_list


def decide_necessary_seldom_basis_tag_list(seldom_basis, common_fact, strategy, seldom_basis_tag_list,num=3):
    necessary_seldom_basis_tag_list = []
    for seldom_b in seldom_basis:
        print("seldom\n", seldom_b)

        all_if_need = {
            "同意": 0,
            "不同意": 0
        }
        counter_seldom_basis_list = create_counter_seldom_basis(seldom_b)
        for counter_seldom_basis in counter_seldom_basis_list:
            common_seldom_fact = common_add_seldom_fact(common_fact, seldom_b, counter_seldom_basis)
            print("common+seldom\n", common_seldom_fact)
            for i in range(num):
                for a, b in seldom_b.items():
                    seldom_key = a
                if_need = check_seldom_basis(common_seldom_fact, strategy, seldom_key, counter_seldom_basis)
                print(if_need)
                all_if_need[if_need] += 1
        print(str(all_if_need))
        if all_if_need["不同意"] >= 1:  # (all_if_need["不同意"] + all_if_need["同意"]) // 3:
            this_tag = find_basis(seldom_b, seldom_basis_tag_list)
            necessary_seldom_basis_tag_list.append(this_tag)
    return necessary_seldom_basis_tag_list


if __name__ == "__main__":
    similar_event_type = "论文抄袭造假"
    similar_strategy = "停职、处分"
    with open("see_classified_list.json", "r", encoding="utf-8") as f:
        see_classified = json.load(f)
    common_basis, seldom_basis = split_by_avg(see_classified)
    # print("common:\n", common_basis)
    # print("seldom:\n", seldom_basis)
    # """
    #     对于“多数情况”，通过多个角度对其概括，然后生成一条涵盖这些概括角度的事件句block1
    # """
    # common_basis_tag_list = create_basis_tag_list(common_basis)
    # with open("common_basis_tag.json", "w", encoding="utf-8") as f:
    #     json.dump(common_basis_tag_list, f, ensure_ascii=False, indent=4)
    with open("common_basis_tag.json", "r", encoding="utf-8") as f:
        common_basis_tag_list = json.load(f)
    common_fact = create_common_fact(common_basis_tag_list)
    """
        对于“少数情况”，也是先概括
    """
    # seldom_basis_tag_list = create_basis_tag_list(seldom_basis)
    # with open("seldom_basis_tag_list.json", "w", encoding="utf-8") as f:
    #     json.dump(seldom_basis_tag_list, f, ensure_ascii=False, indent=4)
    with open("seldom_basis_tag_list.json", "r", encoding="utf-8") as f:
        seldom_basis_tag_list = json.load(f)
    """
        然后生成可能的多个反义/不同角度的概括
        然后分别补充进入fact中
    """
    num = 3
    necessary_seldom_basis_tag_list = decide_necessary_seldom_basis_tag_list(seldom_basis, common_fact, similar_strategy, seldom_basis_tag_list, num)
    with open("necessary_seldom_basis_tag.json", "w", encoding="utf-8") as f:
        json.dump(necessary_seldom_basis_tag_list, f, ensure_ascii=False, indent=4)


