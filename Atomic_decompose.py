# -*- coding: utf-8 -*-
from openai import OpenAI
from typing import List, Dict, Optional, Any
import json
import os
import re
from collections import defaultdict, Counter
import pandas as pd
import ast
from PROMPT import *

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

def qa_message_openai(prompt: str, answer1: str, prompt_2: str, tem = 1) -> str:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful planner."},
            {"role": "user", "content": prompt},
            {"role": "system", "content": answer1},
            {"role": "user", "content": prompt_2}
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




def atomic_dec(conditions:dict):
    '''
    :param conditions:
    :return:
    '''
    atom_basis = []
    new_basis = conditions["new_basis"]
    for key, value in new_basis.items():
        prompt = ATOMIC_DEC.format(
            key=key,
            value=value
        )
        print(key,":", value)
        while True:
            try:
                raw_key_value = qa_openai(prompt)
                print("raw_key_value", raw_key_value)
                match = re.search(r"输出\[(.*?)\]", raw_key_value)
                raw_result = match.group(1).strip()
                print("raw_result",raw_result)
                if raw_result == "无需分解":
                    atom_basis.append({key:value})
                    break
                else:
                    parts = []
                    for seg in raw_result.split(";"):
                        seg = seg.strip()
                        if not seg:
                            continue
                        if ":" in seg:
                            key, val = seg.split(":", 1)
                            parts.append({key.strip(): val.strip()})
                        else:

                            raise ValueError(f"分片格式错误: {seg}")
                    atom_basis.extend(parts)
                    break
            except Exception as e:
                print("匹配失败，重试中:", e)

    return atom_basis

def classify_basis(basis_list):
    show_list = []
    for basis in basis_list:
        for key, value in basis.items():
            if len(show_list) == 0:  # 如果是刚开始，则直接加入
                show_list.append({key:[value]})
            else:
                str_show_list = ""
                i= 0
                for a_show in show_list:
                    for a_key, a_val in a_show.items():
                        a_str = "类别" + str(i) + ":\t" + a_key + "\t包含同类内容:\t" + str(a_val) + "\n"
                        str_show_list += a_str
                        i += 1
                feedback = None  # 保存“不同意”的专家反馈

                while True:
                    try:
                        base_prompt = CLASSIFY.format(
                            classified_list=str_show_list,
                            key=key,
                            value=value,
                            list_index=str(i - 1)
                        )

                        if feedback:
                            prompt1 = f"{base_prompt}\n注意：请结合专家反馈意见『{feedback}』重新判断分类。"
                        else:
                            prompt1 = base_prompt

                        print("prompt1:", prompt1)
                        if_classify = qa_openai(prompt1)
                        print("if_classify:", if_classify)


                        if "新建" in if_classify:
                            prompt2 = CHECK_NEW.format(
                                classified_list=str_show_list,
                                key=key,
                                value=value
                            )
                            if_new = qa_message_openai(prompt1, if_classify, prompt2)
                            print("if_new:", if_new)

                            if if_new == "同意":
                                # 确认新建
                                show_list.append({key: [value]})
                                break
                            else:
                                # 提取“不同意+[意见]XXX”里的意见
                                match = re.search(r"不同意(.*)", if_new)
                                if match:
                                    feedback = match.group(1).strip()
                                    print("extracted feedback:", feedback)
                                    continue  # 回到 while 顶部，重新生成 prompt1（带反馈版）
                                else:
                                    print("Warning: if_new 格式不符合预期:", if_new)
                                    continue
                        elif "归类" in if_classify and re.search(r"\d+", if_classify):
                            match = re.search(r"[+]?(\d+)", if_classify)
                            if match:
                                index = int(match.group(1))
                                prompt3 = CHECK_ADD.format(
                                    classified_list=str_show_list,
                                    key=key,
                                    value=value,
                                    index=str(index)
                                )
                                if_add = qa_message_openai(prompt1, if_classify, prompt3)
                                if if_add == "同意":
                                    for show_key, show_value in show_list[index].items():
                                        show_value.append(value)
                                    break
                                else:
                                    match = re.search(r"不同意(.*)", if_add)
                                    if match:
                                        feedback = match.group(1).strip()
                                        print("extracted feedback:", feedback)
                                        continue  # 回到 while 顶部，重新生成 prompt1（带反馈版）
                                    else:
                                        print("Warning: if_new 格式不符合预期:", if_add)
                                        continue
                            else:
                                raise ValueError(f"无法提取 index: {if_classify}")

                        else:
                            raise RuntimeError(f"if_classify 格式异常: {if_classify}")
                    except Exception as e:
                        print("ERROR:", e)
                        continue
    return show_list


def split_by_avg(data):
    """
    输入: data (list)，其中每个元素是 dict，dict 的 value 是 list
    输出: (avg_length, more_than_avg, less_or_equal_avg)
    """
    # 计算每个 dict 中 value 的长度
    value_lengths = [len(list(d.values())[0]) for d in data]

    # 计算平均数
    avg_length = sum(value_lengths) / len(value_lengths)

    # 按平均数划分
    more_than_avg = [d for d in data if len(list(d.values())[0]) >= avg_length]
    less_or_equal_avg = [d for d in data if len(list(d.values())[0]) < avg_length]

    return more_than_avg, less_or_equal_avg



if __name__ == "__main__":
    # with open("学术不端-停职similar_event.json", "r", encoding="utf-8") as f:
    #     data = json.load(f)
    # basis_list = []
    # for file in similar_events:
    #     atom_basis_list = atomic_dec(data[0][file])
    #     for a in atom_basis_list:
    #         basis_list.append(a)
    # with open("see.json", "w", encoding="utf-8") as f:
    #     json.dump(basis_list, f, ensure_ascii=False, indent=4)
    with open("see.json", "r", encoding="utf-8") as f:
        see_dict = json.load(f)
    classified_list = classify_basis(see_dict)
    with open("see_classified_list.json", "w", encoding="utf-8") as f:
        json.dump(classified_list, f, ensure_ascii=False, indent=4)

