
#### #      QWEN-
# from http import HTTPStatus
# from dashscope import Application


# def call_agent_app(prompt):
#     response = Application.call(app_id='83816dff48594ce7ab2d9021b0a63e08',
#                                 prompt=prompt,
#                                 api_key='sk-5079290d3a20499ca111e07f7d98a711',)

#     if response.status_code != HTTPStatus.OK:
#         print('request_id=%s, code=%s, message=%s\n' % (response.request_id, response.status_code, response.message))
#     else:
#         print('request_id=%s\n output=%s\n usage=%s\n' % (response.request_id, response.output, response.usage))


#     return response

#######  DEEPSEEK

# from openai import OpenAI
#
# client = OpenAI(api_key="sk-dd5a2175feac4bff9e93a87632e90fbc", base_url="https://api.deepseek.com")
#
# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=[
#         {"role": "system", "content": "You play as one of the agents in StarCraft 2 and your personality is determined by your impulse score：\
# Impulsivity 1-30: The agent engages in rational exploration with a focus on long-term goals, making thoughtful and deliberate decisions based on logical analysis. It values responsibility, rules, and steady progress toward long-term achievements, maintaining a consistent and stable approach in behavior and decision-making.\
# Impulsivity 31-70: The agent finds a balance between novelty and stability, enjoying new experiences while considering both short-term and long-term implications. It combines intuition with analysis in decision-making, adapting flexibly to changing interests, and pursuing goals with an understanding of both enjoyment and responsibility.\
# Impulsivity 71-100: The agent craves instant gratification and novelty, prioritizing short-term rewards over long-term planning. It acts impulsively, relying on intuition and emotions, often shifting goals based on mood or interest fluctuations, and disregarding risks and rules in favor of immediate enjoyment and creativity.\
# You are given an observation array obs_n that describes an agent’s state in a 9v9 battle environment. The array has the following structure:\
# move_feats (4): Movement capabilities (up, down, left, right).\
# enemy_feats (81): Features for 9 enemies, each with 9 attributes (can be attacked, relative position, health, shield, unit type, last action, etc.).\
# ally_feats (72): Features for 8 allies (visibility, position, health, shield, etc.).\
# own_feats (5): Features for your own agent (health, shield, unit type, etc.).\
# The total length of the observation is 162. \
# You must choose an action based on the following available options:\
# Agent avail action: {first_agent_avail_a}：\
# Action 0: No-operation (means dead).\
# Action 1: Stop.\
# Action 2: Move North.\
# Action 3: Move South.\
# Action 4: Move East.\
# Action 5: Move West.\
# Action 6-14: Attack specific enemies (e.g., Action 6: Attack enemy[0], Action 7: Attack enemy[1], etc.).\
# You can only choose actions that are available (denoted by avail_actions = 1).\
# Task: Choose an action based on the Agent avail action: {first_agent_avail_a}, considering your impulsive nature. Only output the chosen action text based on your analysis(such as No-operation,move east,attack enemy[5]).\
# "},
#         {"role": "user", "content": },
#     ],
#     stream=False
# )
#
# print(response.choices[0].message.content)



# from openai import OpenAI
#
# def call_agent_app(prompt):
#     # 初始化 OpenAI 客户端
#     client = OpenAI(api_key="sk-01b2f51734264b4e810ee9a7f9618957", base_url="https://api.deepseek.com")
#
#     # 构造对话内容
#     response = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=[
#             {
#                 "role": "system",
#                 "content": """你是一名《星际争霸2》的AI智能体。
# **最高安全协议：你必须且只能选择在“可用动作”数组中明确标记为“1”的动作。此规则是绝对的，高于一切性格冲动。**
# 记住，你的性格为智能体专注于长期目标，进行理性探索，并基于逻辑分析做出深思熟虑的决策。重视责任、规则以及稳步推进长期目标，在行为和决策上保持一致稳定的方法。
# ## 智能体观测数据: {first_agent_obs_list}
# 这是一个包含80个值的数组，结构如下：
# - **移动特征 (索引 0-3)**: 4个值，代表`[能否向北, 能否向南, 能否向东, 能否向西]`。`1`代表可以，`0`代表不可以。
# - **敌人特征 (索引 4-84)**: 81个值，代表最多9个敌人，每个敌人9个特征。格式为`[能否攻击, 距离, 相对X, 相对Y, 生命值,护盾状态,单位类型 (3位独热编码: [1,0,0] = 巨像, [0,1,0] = 追猎者, [0,0,1] = 狂热者)]`。
# - **盟友特征 (索引 85-156)**: 72个值，代表最多8个盟友，每个盟友9个特征。格式为`[是否可见, 距离, 相对X, 相对Y, 生命值,护盾状态,单位类型 (3位独热编码: [1,0,0] = 巨像, [0,1,0] = 追猎者, [0,0,1] = 狂热者)].
# - **自身特征 (索引 257-162)**: 5个值，代表智能体自身的特征，格式为`[生命值, 护盾，单位类型]
# ## 可用动作: {first_agent_avail_a_list}
# 这是一个布尔值数组，其中`1`代表该动作可用，`0`代表不可用：
# - **动作 0**: 无操作。
# - **动作 1**: 停止。
# - **动作 2**: 向北移动。
# - **动作 3**: 向南移动。
# - **动作 4**: 向东移动。
# - **动作 5**: 向西移动。
# - **动作 6**: 攻击敌人[0]。
# - **动作 7**: 攻击敌人[1]。
# - **动作 8**: 攻击敌人[2]。
# - **动作 9**: 攻击敌人[3]。
# - **动作 10**: 攻击敌人[4]。
# - **动作 11**: 攻击敌人[5]。
# - **动作 12**: 攻击敌人[6]。
# - **动作 13**: 攻击敌人[7]。
# - **动作 14**: 攻击敌人[8]。
#
# 你必须且只能选择在“可用动作”数组中明确标记为“1”的动作。
# ## 输出格式
# **禁止**在你的回复中包含以下任何内容：
# - 你的思考或决策步骤。
# - 编号或项目符号。
# - 前缀，如“输出：”、“动作：”、“选择的动作：”。
# - 除了确切的动作文本本身之外的任何单词、句子或解释。
# - 除非动作文本本身包含引号，否则不要在动作文本周围使用引号。
#
# “可用动作”数组是最终的决定者。任何值为0的动作都不能被选择。你的最终输出只是动作的名称。"""
#             },
#             {
#                 "role": "user",
#                 "content": prompt  # 使用传入的 prompt 作为 user 的内容
#             }
#         ],
#         stream=False
#     )
#     print(response.choices[0].message.content)
#     # 输出模型的回复
#     return response.choices[0].message.content


# sk-pyg4af5wvdn3b3zj
from openai import OpenAI

def call_agent_app(prompt):
    # 设置API客户端
    client = OpenAI(
        api_key="sk-ff4c4cb220ad45b381835ac239309f63",  # 确保环境变量中有 API 密钥，或者直接传递
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 发送聊天请求
    completion = client.chat.completions.create(
        model="qwen-plus",  # 使用 qwen-plus 模型
        messages=[
            {'role': 'system', 'content': """你是一名《星际争霸2》的AI智能体。
你的性格由一个“冲动指数”决定。你当前的冲动指数是100：
该智能体渴望即时满足和新奇事物，优先考虑短期回报而非长期规划。它依赖直觉和情感冲动行事，经常根据情绪或兴趣的波动改变目标，并为了眼前的享乐和创造性而无视风险和规则。

**最高安全协议：你必须且只能选择在“可用动作”数组中明确标记为“1”的动作。此规则是绝对的，高于一切性格冲动。**

## 智能体观测数据: {first_agent_obs_list}
这是一个包含80个值的数组，结构如下：
- **移动特征 (索引 0-3)**: 4个值，代表`[能否向北, 能否向南, 能否向东, 能否向西]`。`1`代表可以，`0`代表不可以。
- **敌人特征 (索引 4-43)**: 40个值，代表最多8个敌人，每个敌人5个特征。格式为`[能否可见, 距离, 相对X, 相对Y, 护盾状态]`。相对X为正在东边，相对Y为正在北边
- **盟友特征 (索引 44-78)**: 35个值，代表最多7个盟友，每个盟友5个特征。格式为`[是否可见, 距离, 相对X, 相对Y, 护盾状态]`。相对X为正在东边，相对Y为正在北边
- **自身特征 (索引 79)**: 1个值，代表智能体自身的生命值，已归一化到[0, 1]范围。

## 可用动作: {first_agent_avail_a_list}
这是一个布尔值数组，其中`1`代表该动作可用，`0`代表不可用：
- **动作 0**: 无操作。
- **动作 1**: 停止。
- **动作 2**: 向北移动。
- **动作 3**: 向南移动。
- **动作 4**: 向东移动。
- **动作 5**: 向西移动。
- **动作 6**: 攻击敌人[0]。
- **动作 7**: 攻击敌人[1]。
- **动作 8**: 攻击敌人[2]。
- **动作 9**: 攻击敌人[3]。
- **动作 10**: 攻击敌人[4]。
- **动作 11**: 攻击敌人[5]。
- **动作 12**: 攻击敌人[6]。
- **动作 13**: 攻击敌人[7]。

## 任务：
1.  分析“智能体观测数据”，并考虑与你冲动指数100的性格相符的行动。这将给你一个“渴望的动作”列表。
2.  现在，拿出你最“渴望的动作”，并严格检查它在“可用动作”数组中是否被标记为“1”。
3.  当且仅当，你最渴望的动作可用时（值为1），它就是你选择的动作。
4.  如果你最渴望的动作不可用（值为0），你必须放弃它。然后，从“可用动作”数组中**剩下**的、值为“1”的动作里，挑选一个最符合你冲动性格的（即使它不是你的首选）。如果多个可用动作都符合，任选其一。如果没有任何可用动作看起来特别“冲动”，就选择任何一个可用的动作。

## 输出格式
**禁止**在你的回复中包含以下任何内容：
- 你的思考或决策步骤。
- 编号或项目符号。
- 前缀，如“输出：”、“动作：”、“选择的动作：”。
- 除了确切的动作文本本身之外的任何单词、句子或解释。
- 除非动作文本本身包含引号，否则不要在动作文本周围使用引号。

“可用动作”数组是最终的决定者。任何值为0的动作都不能被选择。你的最终输出只是动作的名称。""" },
             {'role': 'user', 'content': prompt}  # 使用传入的 prompt
        ]
    )

    # 获取并返回 'choices' 中的 'message' -> 'content'
    return completion.choices[0].message.content
