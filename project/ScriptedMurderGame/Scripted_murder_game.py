import os
import random
import sys
import io
from typing import TypedDict, Optional, List
from dotenv import load_dotenv

# 设置 stdout 编码为 UTF-8，解决 Windows 终端编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ================== 初始化环境变量 & LLM ==================
load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.7
)

# ================== 辅助函数 ==================
def wait_for_y():
    """等待用户输入 'y' 继续"""
    while True:
        user_input = input("请输入 'y' 继续：").strip().lower()
        if user_input == 'y':
            break
        print("❌ 请输入 'y'")

# ================== 定义全局状态 ==================
class GameState(TypedDict):
    script: dict
    truth: str
    players: List[dict]
    user_player_idx: int
    game_phase: str
    clues: List[str]
    player_clues: dict
    introductions: dict
    clue_descriptions: dict
    votes: dict
    game_result: str

# ================== 节点1：剧本生成智能体 ==================
def generate_script_node(state: GameState) -> dict:
    print("\n" + "="*60)
    print("📝 节点1：剧本生成智能体")
    print("="*60)

    prompt = ChatPromptTemplate.from_template(
        """请生成一个剧本杀剧本，包含以下内容：

【剧本要求】
- 场景：现代都市豪宅
- 死者：李明远，45岁，科技公司CEO
- 嫌疑人：4位（请设计详细身份和背景故事）
- 主题：商业阴谋与复仇

【输出格式】（必须严格按此格式输出）
【案件背景】
简要描述案发情况，包括时间、地点、死者状态等。

【嫌疑人信息】
1号：姓名：[名字]
   身份：[详细身份描述]
   秘密：[需要隐藏的秘密]
   行动：[案发当天行动轨迹]
2号：姓名：[名字]
   身份：[详细身份描述]
   秘密：[需要隐藏的秘密]
   行动：[案发当天行动轨迹]
3号：姓名：[名字]
   身份：[详细身份描述]
   秘密：[需要隐藏的秘密]
   行动：[案发当天行动轨迹]
4号：姓名：[名字]
   身份：[详细身份描述]
   秘密：[需要隐藏的秘密]
   行动：[案发当天行动轨迹]

【线索列表】
线索1：[线索内容]
线索2：[线索内容]
线索3：[线索内容]
线索4：[线索内容]

【案件真相】
详细描述凶手身份、作案动机、作案手法和过程。

请用中文输出，格式清晰，每项内容都要完整。
        """
    )

    chain = prompt | llm
    
    # 尝试生成剧本，最多重试3次
    max_retries = 3
    script_data = None
    
    for attempt in range(max_retries):
        print(f"\n正在生成剧本... (尝试 {attempt + 1}/{max_retries})")
        result = chain.invoke({})
        content = result.content
        
        # 打印生成的原始内容供调试
        print("\n📄 生成的剧本内容：")
        print("-" * 50)
        print(content[:1000] + "..." if len(content) > 1000 else content)
        print("-" * 50)
        
        script_data = parse_script(content)
        
        if script_data:
            print("✅ 剧本生成成功！")
            break
        else:
            print(f"❌ 剧本解析失败，正在重试...")
    
    if not script_data:
        raise Exception("无法生成有效的剧本，请检查提示词格式或重试")

    return {
        "script": script_data,
        "truth": script_data.get("truth", ""),
        "clues": script_data.get("clues", []),
        "players": [],
        "user_player_idx": 0,
        "game_phase": "script_generated",
        "player_clues": {},
        "introductions": {},
        "clue_descriptions": {},
        "votes": {},
        "game_result": ""
    }

def parse_script(content: str) -> dict:
    """解析生成的剧本内容，完全由LLM生成，不使用备用数据"""
    lines = content.split('\n')
    script = {
        "background": "",
        "suspects": [],
        "clues": [],
        "truth": ""
    }

    current_section = None
    current_suspect = None
    suspect_count = 0

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line.startswith('【') and line.endswith('】'):
            if "案件背景" in line:
                current_section = "background"
            elif "嫌疑人信息" in line:
                current_section = "suspects"
                current_suspect = None
            elif "线索列表" in line:
                current_section = "clues"
            elif "案件真相" in line:
                current_section = "truth"
            continue

        if current_section == "background":
            script["background"] += line + "\n"
        elif current_section == "suspects":
            # 处理格式：1号：姓名：张三 或 1. 姓名 —— 身份
            if (line.startswith("1号") or line.startswith("2号") or 
                line.startswith("3号") or line.startswith("4号") or
                line.startswith("1.") or line.startswith("2.") or
                line.startswith("3.") or line.startswith("4.")):
                
                if current_suspect:
                    script["suspects"].append(current_suspect)
                
                suspect_count += 1
                current_suspect = {
                    "name": "",
                    "role": "",
                    "secret": "",
                    "action": ""
                }
                
                # 尝试提取姓名和身份
                # 格式1：1号：姓名：张三
                if "姓名：" in line:
                    parts = line.split("姓名：", 1)[1].strip()
                    # 继续检查是否有身份信息
                    if "身份：" in parts:
                        name_part, role_part = parts.split("身份：", 1)
                        current_suspect["name"] = name_part.strip()
                        current_suspect["role"] = role_part.strip()
                    else:
                        current_suspect["name"] = parts
                # 格式2：1. 姓名 —— 身份描述
                elif "——" in line:
                    name_part, role_part = line.split("——", 1)
                    # 移除序号部分
                    name_part = name_part.replace("1.", "").replace("2.", "").replace("3.", "").replace("4.", "").strip()
                    name_part = name_part.replace("1号：", "").replace("2号：", "").replace("3号：", "").replace("4号：", "").strip()
                    current_suspect["name"] = name_part.strip()
                    current_suspect["role"] = role_part.strip()
                else:
                    # 尝试其他方式提取姓名
                    parts = line.split("：")
                    for part in parts:
                        p = part.strip()
                        if p and not p.endswith("号") and not p.endswith("."):
                            current_suspect["name"] = p
                            break
            elif current_suspect:
                if line.startswith("身份："):
                    current_suspect["role"] = line.replace("身份：", "").strip()
                elif line.startswith("秘密："):
                    current_suspect["secret"] = line.replace("秘密：", "").strip()
                elif line.startswith("行动："):
                    current_suspect["action"] = line.replace("行动：", "").strip()
                elif line.startswith("姓名："):
                    current_suspect["name"] = line.replace("姓名：", "").strip()
                elif current_suspect["role"] and not current_suspect["secret"]:
                    current_suspect["role"] += " " + line
        elif current_section == "clues":
            if line.startswith("线索"):
                parts = line.split("：", 1)
                clue_text = parts[1].strip() if len(parts) > 1 else line
                script["clues"].append(clue_text)
        elif current_section == "truth":
            script["truth"] += line + "\n"

    if current_suspect:
        script["suspects"].append(current_suspect)

    # 验证解析结果，不完整则返回 None
    # 确保完全由 LLM 生成，不使用任何固定备用数据
    if (len(script["suspects"]) < 4 or 
        len(script["clues"]) < 4 or 
        not script["truth"].strip() or 
        not script["background"].strip()):
        return None

    # 确保每个嫌疑人都有姓名
    for suspect in script["suspects"]:
        if not suspect["name"] or suspect["name"] == "姓名":
            return None

    return script

# ================== 节点2：DM智能体介绍案件 ==================
def dm_intro_node(state: GameState) -> dict:
    print("\n" + "="*60)
    print("🎭 节点2：DM智能体介绍案件")
    print("="*60)

    print("\n欢迎各位来到「推理剧本杀」游戏！")
    print("我是今天的主持人DM。")
    print("今天我们要破解一起发生在豪宅中的谋杀案。\n")

    print("📖 案件背景：")
    print(state["script"]["background"])

    print("\n🕵️ 嫌疑人名单（共4位）：")
    for i, suspect in enumerate(state["script"]["suspects"], 1):
        print(f"{i}. {suspect['name']} —— {suspect['role']}")

    print("\n⚠️ 游戏规则：")
    print("1. 自我介绍：每位玩家介绍身份和案发当天行动")
    print("2. 线索抽取：每位玩家获得一条线索")
    print("3. 线索讨论：分享线索并进行推理")
    print("4. 投票环节：投票选出凶手并说明理由")
    print("5. 真相揭示：DM公布真凶和作案手法\n")

    wait_for_y()

    return {"game_phase": "ready_for_role"}

# ================== 节点3：身份分配智能体 ==================
def role_assignment_node(state: GameState) -> dict:
    print("\n" + "="*60)
    print("🎲 节点3：身份分配智能体")
    print("="*60)

    suspects = state["script"]["suspects"].copy()
    random.shuffle(suspects)

    players = []
    ai_names = ["智能体AI 1", "智能体AI 2", "智能体AI 3"]
    for i, suspect in enumerate(suspects):
        players.append({
            "id": i,
            "name": suspect["name"],
            "role": suspect["role"],
            "secret": suspect["secret"],
            "action": suspect["action"],
            "is_user": i == 0,
            "ai_name": ai_names[i-1] if i > 0 else ""
        })

    print("\n🎭 身份分配结果：")
    print("-" * 50)
    for player in players:
        if player["is_user"]:
            print(f"👉 {player['name']} —— {player['role']} （你）")
        else:
            print(f"   {player['name']} —— {player['role']} （{player['ai_name']}）")
    print("-" * 50)

    wait_for_y()

    return {
        "players": players,
        "user_player_idx": 0,
        "game_phase": "roles_assigned"
    }

# ================== 节点4：DM智能体分配剧本 ==================
def dm_distribute_script_node(state: GameState) -> dict:
    print("\n" + "="*60)
    print("📋 节点4：DM智能体分配剧本")
    print("="*60)

    user_player = state["players"][state["user_player_idx"]]

    print(f"\n🎭 你的角色：【{user_player['name']}】")
    print("=" * 50)
    print(f"身份：{user_player['role']}")
    print(f"\n📝 案发当天你的行动：")
    action_lines = user_player['action'].split("，")
    for i, line in enumerate(action_lines, 1):
        print(f"   {i}. {line.strip()}")
    print(f"\n🔒 你的秘密（请谨慎暴露）：")
    print(user_player['secret'])
    print("=" * 50)

    wait_for_y()

    return {"game_phase": "self_intro"}

# ================== 节点5：玩家用户自我介绍 ==================
def user_intro_node(state: GameState) -> dict:
    print("\n" + "="*60)
    print("🎤 节点5：玩家自我介绍环节")
    print("="*60)

    user_player = state["players"][state["user_player_idx"]]

    print(f"\n🎙️ 轮到你了，【{user_player['name']}】！")
    print("请介绍你的身份以及案发当天的行动：")
    intro = input("> ")

    state["introductions"][user_player["id"]] = intro

    print(f"\n【{user_player['name']}】（你）说：{intro}")

    wait_for_y()

    return {"game_phase": "ai_intro_1"}

# ================== 节点6：玩家智能体1发言 ==================
def ai_intro_1_node(state: GameState) -> dict:
    print("\n" + "="*60)
    print("🤖 节点6：智能体AI 1发言")
    print("="*60)

    player = state["players"][1]

    prompt = ChatPromptTemplate.from_template(
        """你是剧本杀玩家{name}，身份是{role}。
你的秘密是：{secret}
案发当天你的行动：{action}

请做一个自我介绍，介绍你的身份和案发当天的行动。
注意：可以适当隐藏你的秘密，但要保持言行一致。
语言要自然，符合角色身份。
        """
    )

    chain = prompt | llm
    result = chain.invoke({
        "name": player["name"],
        "role": player["role"],
        "secret": player["secret"],
        "action": player["action"]
    })

    intro = result.content
    state["introductions"][player["id"]] = intro

    print(f"\n🎙️ 【{player['name']}】（{player['ai_name']}）说：{intro}")

    wait_for_y()

    return {"game_phase": "ai_intro_2"}

# ================== 节点7：玩家智能体2发言 ==================
def ai_intro_2_node(state: GameState) -> dict:
    print("\n" + "="*60)
    print("🤖 节点7：智能体AI 2发言")
    print("="*60)

    player = state["players"][2]

    prompt = ChatPromptTemplate.from_template(
        """你是剧本杀玩家{name}，身份是{role}。
你的秘密是：{secret}
案发当天你的行动：{action}

请做一个自我介绍，介绍你的身份和案发当天的行动。
注意：可以适当隐藏你的秘密，但要保持言行一致。
语言要自然，符合角色身份。
        """
    )

    chain = prompt | llm
    result = chain.invoke({
        "name": player["name"],
        "role": player["role"],
        "secret": player["secret"],
        "action": player["action"]
    })

    intro = result.content
    state["introductions"][player["id"]] = intro

    print(f"\n🎙️ 【{player['name']}】（{player['ai_name']}）说：{intro}")

    wait_for_y()

    return {"game_phase": "ai_intro_3"}

# ================== 节点8：玩家智能体3发言 ==================
def ai_intro_3_node(state: GameState) -> dict:
    print("\n" + "="*60)
    print("🤖 节点8：智能体AI 3发言")
    print("="*60)

    player = state["players"][3]

    prompt = ChatPromptTemplate.from_template(
        """你是剧本杀玩家{name}，身份是{role}。
你的秘密是：{secret}
案发当天你的行动：{action}

请做一个自我介绍，介绍你的身份和案发当天的行动。
注意：可以适当隐藏你的秘密，但要保持言行一致。
语言要自然，符合角色身份。
        """
    )

    chain = prompt | llm
    result = chain.invoke({
        "name": player["name"],
        "role": player["role"],
        "secret": player["secret"],
        "action": player["action"]
    })

    intro = result.content
    state["introductions"][player["id"]] = intro

    print(f"\n🎙️ 【{player['name']}】（{player['ai_name']}）说：{intro}")

    wait_for_y()

    return {"game_phase": "dm_clues"}

# ================== 节点9：DM智能体给出线索 ==================
def dm_clues_node(state: GameState) -> dict:
    print("\n" + "="*60)
    print("🔍 节点9：DM智能体给出线索")
    print("="*60)

    print("\n📋 现场发现了以下线索：")
    for i, clue in enumerate(state["clues"], 1):
        print(f"{i}. {clue}")

    wait_for_y()

    return {"game_phase": "clue_assignment"}

# ================== 节点10：线索分配智能体 ==================
def clue_assignment_node(state: GameState) -> dict:
    print("\n" + "="*60)
    print("🎲 节点10：线索分配智能体")
    print("="*60)

    clues = state["clues"].copy()
    random.shuffle(clues)

    player_clues = {}
    for i, player in enumerate(state["players"]):
        if i < len(clues):
            player_clues[player["id"]] = clues[i]
        else:
            player_clues[player["id"]] = "没有找到相关线索"

    print("\n🎁 线索分配结果：")
    print("-" * 50)
    for player in state["players"]:
        if player["is_user"]:
            print(f"👉 【{player['name']}】（你）的线索：")
        else:
            print(f"   【{player['name']}】（{player['ai_name']}）的线索：")
        print(f"     {player_clues[player['id']]}")
    print("-" * 50)

    wait_for_y()

    return {
        "player_clues": player_clues,
        "game_phase": "clue_discussion"
    }

# ================== 节点11：线索讨论环节 ==================
def clue_discussion_node(state: GameState) -> dict:
    print("\n" + "="*60)
    print("💬 节点11：线索讨论环节")
    print("="*60)

    user_player = state["players"][state["user_player_idx"]]
    user_clue = state["player_clues"].get(user_player["id"], "")

    print(f"\n🎙️ 轮到你了，【{user_player['name']}】！")
    print(f"你的线索：{user_clue}")
    print("请描述你对这条线索的看法：")
    user_desc = input("> ")
    state["clue_descriptions"][user_player["id"]] = user_desc
    print(f"\n【{user_player['name']}】（你）说：{user_desc}")

    wait_for_y()

    player1 = state["players"][1]
    clue1 = state["player_clues"].get(player1["id"], "")

    prompt = ChatPromptTemplate.from_template(
        """你是剧本杀玩家{name}，身份是{role}。
你的线索是：{clue}
其他玩家的线索描述：
{other_descriptions}

请描述你对自己线索的看法，可以结合其他玩家的发言进行分析。
语言要自然。
        """
    )

    chain = prompt | llm
    result = chain.invoke({
        "name": player1["name"],
        "role": player1["role"],
        "clue": clue1,
        "other_descriptions": f"{user_player['name']}: {user_desc}"
    })
    desc1 = result.content
    state["clue_descriptions"][player1["id"]] = desc1
    print(f"\n🎙️ 【{player1['name']}】（{player1['ai_name']}）说：{desc1}")

    wait_for_y()

    player2 = state["players"][2]
    clue2 = state["player_clues"].get(player2["id"], "")

    chain = prompt | llm
    result = chain.invoke({
        "name": player2["name"],
        "role": player2["role"],
        "clue": clue2,
        "other_descriptions": f"{user_player['name']}: {user_desc}\n{player1['name']}: {desc1}"
    })
    desc2 = result.content
    state["clue_descriptions"][player2["id"]] = desc2
    print(f"\n🎙️ 【{player2['name']}】（{player2['ai_name']}）说：{desc2}")

    wait_for_y()

    player3 = state["players"][3]
    clue3 = state["player_clues"].get(player3["id"], "")

    chain = prompt | llm
    result = chain.invoke({
        "name": player3["name"],
        "role": player3["role"],
        "clue": clue3,
        "other_descriptions": f"{user_player['name']}: {user_desc}\n{player1['name']}: {desc1}\n{player2['name']}: {desc2}"
    })
    desc3 = result.content
    state["clue_descriptions"][player3["id"]] = desc3
    print(f"\n🎙️ 【{player3['name']}】（{player3['ai_name']}）说：{desc3}")

    wait_for_y()

    return {"game_phase": "voting"}

# ================== 节点12：投票环节 ==================
def voting_node(state: GameState) -> dict:
    print("\n" + "="*60)
    print("🗳️ 节点12：投票环节")
    print("="*60)

    print("\n👤 嫌疑人列表（请根据序号投票）：")
    print("-" * 50)
    for i, player in enumerate(state["players"]):
        if player["is_user"]:
            print(f"{i+1}. 【{player['name']}】 —— {player['role']} （你）")
        else:
            print(f"{i+1}. 【{player['name']}】 —— {player['role']} （{player['ai_name']}）")
    print("-" * 50)

    user_player = state["players"][state["user_player_idx"]]
    print(f"\n🎙️ 轮到你投票了，【{user_player['name']}】！")

    while True:
        try:
            choice = int(input("请选择你怀疑的凶手（输入序号1-4）：")) - 1
            if 0 <= choice < len(state["players"]):
                voted_player = state["players"][choice]
                break
            print("❌ 请输入有效序号（1-4）")
        except ValueError:
            print("❌ 请输入数字")

    print("请说明你的理由：")
    reason = input("> ")
    state["votes"][user_player["id"]] = (voted_player["id"], reason)
    print(f"\n✅ 【{user_player['name']}】（你）投票给：【{voted_player['name']}】")
    print(f"理由：{reason}")

    wait_for_y()

    def get_ai_vote(player, state, intros, discussions):
        """获取AI玩家的投票结果，从LLM输出中解析投票对象"""
        prompt = ChatPromptTemplate.from_template(
            """你是剧本杀玩家{name}，身份是{role}。
你的秘密是：{secret}
你的线索是：{clue}
其他玩家的自我介绍：
{intros}
线索讨论内容：
{discussions}

请分析谁最有可能是凶手，并给出理由。
输出格式必须严格按照以下格式：
投票对象：[嫌疑人姓名]
理由：[你的推理分析]

注意：嫌疑人姓名必须是以下名字之一：{suspect_names}
        """
        )

        suspect_names = ", ".join([p["name"] for p in state["players"]])
        
        chain = prompt | llm
        result = chain.invoke({
            "name": player["name"],
            "role": player["role"],
            "secret": player["secret"],
            "clue": state["player_clues"].get(player["id"], ""),
            "intros": intros,
            "discussions": discussions,
            "suspect_names": suspect_names
        })

        content = result.content
        
        # 解析投票对象
        voted_name = ""
        reason = ""
        
        lines = content.split("\n")
        for line in lines:
            if line.startswith("投票对象：") or line.startswith("投票给："):
                voted_name = line.replace("投票对象：", "").replace("投票给：", "").strip()
            elif line.startswith("理由："):
                reason = line.replace("理由：", "").strip()
        
        # 如果没有找到明确的投票对象，从内容中匹配姓名
        if not voted_name:
            for p in state["players"]:
                if p["name"] in content:
                    voted_name = p["name"]
                    break
        
        # 如果还是没找到，随机选择（作为后备）
        if not voted_name:
            candidates = [p for p in state["players"] if p["id"] != player["id"]]
            voted_name = random.choice(candidates)["name"]
        
        # 找到对应的玩家ID
        voted_id = None
        for p in state["players"]:
            if p["name"] == voted_name:
                voted_id = p["id"]
                break
        
        # 如果还是没找到，随机选择
        if voted_id is None:
            candidates = [p["id"] for p in state["players"] if p["id"] != player["id"]]
            voted_id = random.choice(candidates)
        
        return voted_id, voted_name, content

    intros = "\n".join([f"{state['players'][p_id]['name']}: {intro[:50]}..." for p_id, intro in state["introductions"].items()])
    discussions = "\n".join([f"{state['players'][p_id]['name']}: {desc[:50]}..." for p_id, desc in state["clue_descriptions"].items()])

    player1 = state["players"][1]
    ai1_voted_id, ai1_voted_name, ai1_result = get_ai_vote(player1, state, intros, discussions)
    state["votes"][player1["id"]] = (ai1_voted_id, ai1_result)
    print(f"\n✅ 【{player1['name']}】（{player1['ai_name']}）投票给：【{ai1_voted_name}】")
    print(f"理由：{ai1_result}")

    wait_for_y()

    player2 = state["players"][2]
    ai2_voted_id, ai2_voted_name, ai2_result = get_ai_vote(player2, state, intros, discussions)
    state["votes"][player2["id"]] = (ai2_voted_id, ai2_result)
    print(f"\n✅ 【{player2['name']}】（{player2['ai_name']}）投票给：【{ai2_voted_name}】")
    print(f"理由：{ai2_result}")

    wait_for_y()

    player3 = state["players"][3]
    ai3_voted_id, ai3_voted_name, ai3_result = get_ai_vote(player3, state, intros, discussions)
    state["votes"][player3["id"]] = (ai3_voted_id, ai3_result)
    print(f"\n✅ 【{player3['name']}】（{player3['ai_name']}）投票给：【{ai3_voted_name}】")
    print(f"理由：{ai3_result}")

    wait_for_y()

    return {"game_phase": "reveal_truth"}

# ================== 节点13：DM揭示真相 ==================
def reveal_truth_node(state: GameState) -> dict:
    print("\n" + "="*60)
    print("🎭 节点13：DM揭示真相")
    print("="*60)

    vote_counts = {}
    for p_id, (voted_id, _) in state["votes"].items():
        vote_counts[voted_id] = vote_counts.get(voted_id, 0) + 1

    print("\n📊 投票结果统计：")
    print("-" * 50)
    for p_id, count in vote_counts.items():
        p_name = state["players"][p_id]["name"]
        print(f"   【{p_name}】：{count}票")
    print("-" * 50)

    if vote_counts:
        max_votes = max(vote_counts.values())
        winners = [p_id for p_id, count in vote_counts.items() if count == max_votes]

        print("\n🏆 投票结果：")
        if len(winners) == 1:
            winner_name = state["players"][winners[0]]["name"]
            print(f"   得票最多的是：【{winner_name}】")
        else:
            winner_names = ", ".join([f"【{state['players'][p_id]['name']}】" for p_id in winners])
            print(f"   平票：{winner_names}")
    else:
        winners = []

    print("\n🔍 案件真相：")
    print("=" * 50)
    print(state["truth"])
    print("=" * 50)

    # 从真相中提取真凶名字
    truth_text = state["truth"]
    real_killer = None
    
    # 尝试多种模式提取真凶名字
    # 模式1：凶手是：XXX
    if "凶手是：" in truth_text:
        start = truth_text.find("凶手是：") + 4
        end = truth_text.find("\n", start)
        if end == -1:
            end = len(truth_text)
        real_killer = truth_text[start:end].strip()
    
    # 模式2：真凶是XXX
    if not real_killer and "真凶是" in truth_text:
        start = truth_text.find("真凶是") + 3
        end = truth_text.find("\n", start)
        if end == -1:
            end = len(truth_text)
        real_killer = truth_text[start:end].strip()
    
    # 模式3：凶手：XXX（包含**凶手：** XXX格式）
    if not real_killer:
        if "**凶手：**" in truth_text:
            start = truth_text.find("**凶手：**") + 6
        elif "凶手：**" in truth_text:
            start = truth_text.find("凶手：**") + 4
        elif "凶手：" in truth_text:
            start = truth_text.find("凶手：") + 3
        else:
            start = -1
        
        if start != -1:
            end = truth_text.find("\n", start)
            if end == -1:
                end = len(truth_text)
            real_killer = truth_text[start:end].strip()
    
    # 清理真凶名字中的特殊符号（如**）
    if real_killer:
        real_killer = real_killer.replace("**", "").strip()
    
    # 如果还是没找到，尝试提取所有嫌疑人名字中在真相中出现的第一个
    if not real_killer:
        for p in state["players"]:
            if p["name"] in truth_text:
                real_killer = p["name"]
                break

    result_text = "\n"
    if winners:
        # 获取所有获胜者的名字
        winner_names = [state["players"][winner_id]["name"] for winner_id in winners]
        
        # 判断投票结果是否与真凶一致
        if real_killer and real_killer in winner_names:
            result_text += f"🎉 恭喜！你们成功找出了真凶【{real_killer}】！\n"
        else:
            result_text += "❌ 很遗憾，凶手猜错了！"
            if real_killer:
                result_text += f" 真凶是【{real_killer}】！\n"
            else:
                result_text += "\n"
    else:
        result_text += "⚠️ 没有有效投票\n"

    print("\n🔒 各玩家的秘密：")
    print("-" * 50)
    for player in state["players"]:
        if player["is_user"]:
            print(f"👉 【{player['name']}】（你）的秘密：{player['secret']}")
        else:
            print(f"   【{player['name']}】（{player['ai_name']}）的秘密：{player['secret']}")
    print("-" * 50)

    print(result_text)

    return {"game_result": result_text}

# ================== 构建 LangGraph ==================
def build_game_graph():
    graph = StateGraph(GameState)

    graph.add_node("generate_script", generate_script_node)
    graph.add_node("dm_intro", dm_intro_node)
    graph.add_node("role_assignment", role_assignment_node)
    graph.add_node("dm_distribute_script", dm_distribute_script_node)
    graph.add_node("user_intro", user_intro_node)
    graph.add_node("ai_intro_1", ai_intro_1_node)
    graph.add_node("ai_intro_2", ai_intro_2_node)
    graph.add_node("ai_intro_3", ai_intro_3_node)
    graph.add_node("dm_clues", dm_clues_node)
    graph.add_node("clue_assignment", clue_assignment_node)
    graph.add_node("clue_discussion", clue_discussion_node)
    graph.add_node("voting", voting_node)
    graph.add_node("reveal_truth", reveal_truth_node)

    graph.add_edge(START, "generate_script")
    graph.add_edge("generate_script", "dm_intro")
    graph.add_edge("dm_intro", "role_assignment")
    graph.add_edge("role_assignment", "dm_distribute_script")
    graph.add_edge("dm_distribute_script", "user_intro")
    graph.add_edge("user_intro", "ai_intro_1")
    graph.add_edge("ai_intro_1", "ai_intro_2")
    graph.add_edge("ai_intro_2", "ai_intro_3")
    graph.add_edge("ai_intro_3", "dm_clues")
    graph.add_edge("dm_clues", "clue_assignment")
    graph.add_edge("clue_assignment", "clue_discussion")
    graph.add_edge("clue_discussion", "voting")
    graph.add_edge("voting", "reveal_truth")
    graph.add_edge("reveal_truth", END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)

    return app

# ================== 主执行函数 ==================
def main():
    print("\n" + "*"*70)
    print("🎭 剧本杀智能体游戏 v2.0")
    print("*"*70)

    app = build_game_graph()

    thread_id = f"scripted_murder_game_{random.randint(1000, 9999)}"

    result = app.invoke(
        {},
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    print("\n" + "*"*70)
    print("🎮 游戏结束")
    print("*"*70)

if __name__ == "__main__":
    main()
