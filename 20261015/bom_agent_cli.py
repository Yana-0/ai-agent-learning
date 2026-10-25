import requests
import json
from dotenv import load_dotenv
import os
import sys
from conversational_agent import ConversationalAgent

# -- 配置 --
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
print("api_key加载成功" if api_key else "未找到api_key")

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "authorization":f"Bearer {api_key}",
    "content_type":"application/json"
}

def get_bom_change(bom_id):
    bom_data = {
        "BOM-1001": "影响物料：螺丝 M3, 钣金件 A12；涉及产线：SMT-2",
        "BOM-1002": "影响物料：电池 5000mAh, 外壳 B 款；涉及产线：组装-3",
    }
    return bom_data.get(bom_id,f"未找到{bom_id}的变更记录")
def get_material_stock(material_name):
    stock_data = {
        "螺丝 M3": "库存充足，现有 5000 个",
        "钣金件 A12": "库存紧张，仅剩 120 个，建议补货",
        "电池 5000mAh": "库存充足，现有 2000 个",
        "外壳 B 款": "无库存，已停产",
    }
    return stock_data.get(material_name,f"未找到{material_name}的库存信息")
# 工具
tools = [
    {
        "type":"function",
        "function":{
            "name":"get_bom_change",
            "description":"根据bom编号查询变更，影响到物料和产线",
            "parameters":{
                "type":"object",
                "properties":{
                    "bom_id":{
                        "type":"string",
                        "description":"bom编号，例如BOM-1001"
                    }
                },
                "required":["bom_id"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"get_material_stock",
            "description":"根据物料名称查询某个物料的当前库存情况",
            "parameters":{
                "type":"object",
                "proterties":{
                    "material_name":{
                        "type":"string",
                        "description":"物料名称，如 螺丝 M3"
                    }
                },
                "required":["material_name"]
            }
        }
    }
]

available_functions = {
    "get_bom_change":get_bom_change,
    "get_material_stock":get_material_stock
}

def run_agent(user_input):
    messages = [
        {"role":"system","content":"你是制造企业的AI助理，回答简洁专业，用中文。"},
        {"role":"user","content":user_input}
    ]

    max_turns = 5
    turn_count = 0
    while turn_count < max_turns:
        turn_count += 1

        # 调用大模型
        data = json.dumps({
            "model":"nvidia/nemotron-3.5-lightning:free",
            "messages":messages,
            "tools":tools,
            "tool_choice":"auto"
        })
        resp = requests.post(url=url, headers=headers, data=data)
        resp = resp.json()["choices"][0]["message"]
        # assistant的msg存入历史消息
        messages.append(resp)

        # 模型中不存在工具调用，返回最终回答
        if not resp.get("tool_calls"):
            return resp["content"]

        # 处理本轮工具调用
        for tc in resp["tool_calls"]:
            func_name = tc["function"]["name"]
            func_args = json.loads(tc["function"]["arguments"])

            if func_name in available_functions:
                func_to_call = available_functions[func_name]
                function_result = func_to_call(**func_args)
            else:
                function_result = f"错误，未找到函数{func_name}"

            #函数调用结果存入历史消息
            messages.append({
                "role":"tool",
                "tool_call_id":tc["id"],
                "content":function_result
            })

    #超出最大论述，强制总结
    messages.append({
        "role":"user",
        "content":"请基于已有信息给出最终回答。"
    })
    data = json.dumps({
        "model":"nvidia/nemotron-3.5-lightning:free",
        "messages":messages
    })
    resp2 = requests.post(url=url,headers=headers,data=data)
    return resp2.json()["choices"][0]["message"]["content"]

# 假设有个 my_tool.py 文件：
# def greet():
#     print("你好")
# # 测试代码
# if __name__ == "__main__":
#     greet()
# ‌直接运行‌ python my_tool.py：__name__ 为 '__main__'，会打印“你好”。
# ‌在另一个文件导入它‌：import my_tool 时，__name__ 变成 'my_tool'，测试代码‌不会执行‌，只导入了 greet 函数。‌‌

# print(f"当前 __name__ 值为: {__name__}")
if __name__ == "__main__":
    # sys.argv 是 Python sys 模块提供的一个‌列表（List）‌，它自动存储了你在命令行运行脚本时输入的所有内容。
    # sys.argv[0]‌：永远是‌脚本文件名‌（例如 bom_agent_cli.py）。
    # sys.argv[1]‌：是你输入的‌第一个参数‌（例如 BOM-1001）。
    # sys.argv[2]‌：是你输入的第二个参数，以此类推。

    # 交互模式：不加参数时进入对话
    if len(sys.argv) < 2:
        print("进入交互模式：输入 exit 退出\n")
        agent = ConversationalAgent(tools, available_functions, api_key)
        while True:
            user_input = input("你：")
            if user_input.lower() in ["exit"]:
                break
            result = agent.chat(user_input)
            print(f"助手：{result}\n")
    # 单次查询模式
    else:
        bom_id = sys.argv[1]
        ques = f"请分析 {bom_id} 的变更影响，并检查受影响物料的库存情况。"

        print(f"正在分析 {bom_id}...\n")
        result = run_agent(ques)
        print(result)

    # 测试用例1
    # python bom_agent_cli.py BOM-1001

    # 测试用例2
    # python bom_agent_cli.py
    # BOM-1001 变更影响什么？
    # 那它的库存够吗？
    # BOM-1002 呢？
    # 刚才说的 BOM-1001 的版本是多少？