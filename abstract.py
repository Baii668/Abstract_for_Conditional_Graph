# -*- coding: utf-8 -*-
"""
之前的步骤是针对每个事件生成一个具体的图谱--对决策生成的解释
此处的终极目的是：对同类事件的相同决策做归纳，形成抽象规律，即多数情况下做出这个决策需要的前置条件是什么（通常情况），另外少数情况下还需要考虑XXX
按理说大约的步骤是：
Step1：根据相同决策把依据的条件都检索出来 ---- 这里初步可以用retrieve.py的class BGERecall
Step2：根据条件的aspect分类，比如这一类都是考虑事件地点和事件后果的，那一类都是考虑责任主体的
Step3：假设都是考虑事件地点，现在有多个具体的条件的value，就应该去概括（归类），此时可以通过多个角度描述，借鉴之前写的反事实的操作
    Step3.1：每组条件-决策对都生成m个概括角度
    Step3.2：从中挑选并集？或者能全部概括所有的角度集合m-
    Step3.3：生成反事实数据，挑选其中n个重要的角度（相当于对前两步做精炼）on
Step4：根据数量上的统计条件是通用的or特殊的
"""
"""
因果推理+组合推理 --> 组合因果推理（CCR），能够推断因果度量如何组合，以及因果量如何在图中传播
此例中考虑，事件A：发生了学术不端事件，事件B：停职撤稿等处罚，现在中间有这m个具体条件，需要总结出更通用的规律条件
"""
import json
import os
from PROMPT import *
from Atomic_decompose import *
from Counterfactual_block import *

# similar_events = [
#     "114_云南高校副院长学位论文抄袭.json",
#     "155_如何看待华中农业大学黄某若教授被课题组十一名成员联合举报学术造假？.json",
#     "386_西安电子科大通报学生毕设代做事件.json",
#     "626_北邮通报学生联名举报导师事件.json"
# ]

if __name__ == "__main__":
    with open("企业群体性伤亡-挂牌督办_events.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    '''
        检索到类似事件后，应该直接生成一个统一的事件类别和应对策略
    '''
    similar_event_type = "企业群体性伤亡"
    similar_strategy = "挂牌督办"
    '''
        对每一个条件原子化，能再分的就再细分
        最后统一存入list
        输出：
        应该是很多的{key:value}
    '''
    basis_list = []
    for file in data:
        for key, value in file.items():
            atom_basis_list = atomic_dec(value)
            for a in atom_basis_list:
                basis_list.append(a)

    '''
        LLM对list中同类条件归类并计数
        就是维护一个新的list，先看key是否等价，然后把value存入
        输出：
        key数量少于上一步，并且某个key中有>=2个value
    '''
    classified_basis_list = classify_basis(basis_list)
    common_basis, seldom_basis = split_by_avg(classified_basis_list)
    '''
        算一个key中value的平均数吧，>=这个数就算“多数情况”，其他算“少数条件”
        对于“多数情况”，通过多个角度对其概括，生成一条涵盖这些概括角度的事件句block1
        对于“少数情况”，分别生成反事实的-block2、3、4...... 
        block1与-block2、3、4依次组合，模型判断是否会更改strategy，有则补充少数情况
        输出：
        event：学术不端类
        strategy：停职处罚
        common_basis:{key:value}
        seldom_basis:{key:value}
    '''
    '''
        先打标签，list中的每一项形如：
            {"事件发生地点": "高校"}
    '''
    common_basis_tag_list = create_basis_tag_list(common_basis)
    seldom_basis_tag_list = create_basis_tag_list(seldom_basis)
    '''
        针对一般条件，生成突发事件摘要（报道）
    '''
    common_fact = create_common_fact(common_basis_tag_list)
    '''
        从所有“少数情况”seldom_basis中选择能影响决策的子集
        对于每个seldom_basis的每个可能的反例，模型投票num次
    '''
    seldom_basis_tag_list = create_basis_tag_list(seldom_basis)
    num = 3
    necessary_seldom_basis_tag_list = decide_necessary_seldom_basis_tag_list(seldom_basis, common_fact,
                                                                             similar_strategy, seldom_basis_tag_list, num)
    '''
        最后，对于类别为similar_event_type的事件similar_event，是否能做出similar_strategy这个决策，要考虑：
        一般的，common_basis_tag_list要符合
        额外的，如果能匹配的上，要考虑necessary_seldom_basis_tag_list是否符合
    '''
    file_name = similar_event_type + "-" + similar_strategy + ".json"
    result = {
        "common_basis": common_basis_tag_list,
        "seldom_basis": necessary_seldom_basis_tag_list,
    }
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)