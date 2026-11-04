import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # 报错：网络环境无法访问 Hugging Face 的服务器（被墙了）

import chromadb
from chromadb.utils import embedding_functions
import json
import requests
from dotenv import load_dotenv
import os

# ---------- 初始化 RAG 引擎 ----------
client = chromadb.PersistentClient(path="../chroma_db")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_collection(name="oa_knowledge_base",embedding_function=embedding_function)
# print(f"集合中的文档数量：{collection.count()}")

def query_knowledge_base(question):
    """
    从企业内部知识库检索相关信息

    参数：
        question: string, 用户的问题

    返回：
        string, 检索到的文档内容
    """
    results = collection.query(query_texts=[question],n_results=3)

    if not results["documents"][0]:
        return "知识库中未找到相关信息"

    retrieved = []
    for doc,metadata,distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        retrieved.append(f"【{metadata['title']}】(相关度：{1-distance:.2f})\n{doc}")
    return "\n\n".join(retrieved)

# ---------- 业务工具函数 ----------
def get_bom_change(bom_id):
    """查询 BOM 变更影响"""
    bom_data = {
        "BOM-1001": "影响物料：螺丝 M3, 钣金件 A12；涉及产线：SMT-2",
        "BOM-1002": "影响物料：电池 5000mAh, 外壳 B 款；涉及产线：组装-3",
    }
    return bom_data.get(bom_id, "未找到该 BOM 的变更记录")

def get_material_stock(material_name):
    """查询物料库存"""
    stock_data = {
        "螺丝 M3": "库存充足，现有 5000 个",
        "钣金件 A12": "库存紧张，仅剩 120 个，建议补货",
        "电池 5000mAh": "库存充足，现有 2000 个",
        "外壳 B 款": "无库存，已停产",
    }
    return stock_data.get(material_name, f"未找到物料 {material_name} 的库存信息")


# ---------- 工具定义（三个工具：查知识库 + 两个业务工具） ----------
tools = [
    {
        "type":"function",
        "function":{
            "name":"query_knowledge_base",
            "description":"查询企业内部知识库，包含 OA 制度、流程规范、管理规定等信息。当用户问到公司制度、流程、规定、规则等问题时使用此工具。",
            "parameters":{
                "type":"object",
                "properties":{
                    "question":{
                        "type":"string",
                        "description":"要查询的问题，尽量完整描述"
                    }
                },
                "required":["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_bom_change",
            "description": "根据 BOM 编号查询变更影响到的物料和产线",
            "parameters": {
                "type": "object",
                "properties": {
                    "bom_id": {
                        "type": "string",
                        "description": "BOM 编号，如 BOM-1001"
                    }
                },
                "required": ["bom_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_material_stock",
            "description": "查询某个物料的当前库存情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "material_name": {
                        "type": "string",
                        "description": "物料名称，如 螺丝 M3"
                    }
                },
                "required": ["material_name"]
            }
        }
    }
]
# 函数名到实际函数的映射
available_functions = {
    "query_knowledge_base": query_knowledge_base,
    "get_bom_change": get_bom_change,
    "get_material_stock": get_material_stock
}

class HybridAgent:
    """
    融合 Agent：既能查知识库，又能调业务工具
    """
    def __init__(self,system_prompt=None):
        load_dotenv()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "authorization":f"Bearer {self.api_key}",
            "content-type":"application/json"
        }
        self.system_prompt = system_prompt or """你是制造企业的 AI 助理，拥有以下能力：
1. 查询企业内部知识库（制度、流程、规定）
2. 查询 BOM 变更影响
3. 查询物料库存
请根据用户问题灵活选择使用哪些能力。如果需要多方面信息才能回答，请逐步查询。"""
        self.messages = [
            {"role":"system","content":self.system_prompt}
        ]

    def chat(self,user_input,max_turns = 8):
        """发送消息，支持多工具调用"""
        self.messages.append({"role":"user","content":user_input})
        turn_count = 0
        while turn_count < max_turns:
            turn_count += 1
            print(f"\n--- 第 {turn_count} 轮思考 ---")

            data = json.dumps({
                "model":"nvidia/nemotron-3.5-lightning:free",
                "messages":self.messages,
                "tools":tools,
                "tool_choice":"auto"
            })
            resp = requests.post(url=self.url,headers=self.headers,data=data)
            resp = resp.json()["choices"][0]["message"]
            self.messages.append(resp)

            if not resp.get("tool_calls"):
                print("模型直接回答，不再调工具")
                return resp["content"]

            for tc in resp["tool_calls"]:
                func_name = tc["function"]["name"]
                func_args = json.loads(tc["function"]["arguments"])

                print(f"→ 调用工具：{func_name}")
                print(f"  参数：{func_args}")

                if func_name in available_functions:
                    try:
                        func_to_call = available_functions[func_name]
                        func_result = func_to_call(**func_args)
                        print(f"结果：{func_result[:80]}...")
                    except Exception as e:
                        func_result = f"函数执行出错：{str(e)}"
                        print(f"出错：{e}")
                else:
                    func_result = f"错误：未找到函数{func_name}"

                self.messages.append({
                    "role":"tool",
                    "tool_call_id":tc["id"],
                    "content":str(func_result)
                })

        # 超出最大轮数
        self.messages.append({"role":"user","content":"请基于已有信息给出最终回答。"})
        data = json.dumps({
            "model":"nvidia/nemotron-3.5-lightning:free",
            "messages":self.messages
        })
        resp = requests.post(url=self.url,headers=self.headers,data=data)
        return resp.json()["choices"][0]["message"]["content"]

    def reset(self):
        """重置对话"""
        self.messages=[{"role":"system","content":self.system_prompt}]


# ---------- 测试 ----------------
if __name__ == "__main__":
    agent = HybridAgent()
    test_cases = [
        # 测试 1：只需要查知识库
        "离职流程是怎样的？",

        # 测试 2：只需要调业务工具
        "BOM-1001 变更影响了哪些物料？",

        # 测试 3：需要查知识库 + 调工具（核心测试）
        "BOM-1001 发生了变更，按照公司制度，我需要通知哪些部门？",

        # 测试 4：复杂场景，多步推理
        "BOM-1002 变更了，受影响的物料库存够吗？如果不够，按照公司规定应该怎么做？",
    ]

    for i,question in enumerate(test_cases,1):
        print(f"\n\n{'='*60}")
        print(f"测试 {i}：{question}")
        print(f"{'='*60}")

        answer = agent.chat(question)
        print(f"\n最终回答：{answer}")

        agent.reset()  # 每个测试用例结束后重置对话
        print("\n" + "-"*40)