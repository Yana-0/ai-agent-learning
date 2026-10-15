import requests
import json
from dotenv import load_dotenv
import os

class ConversationalAgent:
    """
    带记忆的对话式 Agent
    """

    def __init__(self, tools, available_functions, api_key, system_prompt=None):
        self.tools = tools
        self.available_functions = available_functions

        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-type": "application/json"
        }
        self.system_prompt = system_prompt or "你是制造企业的 AI 助理，回答简洁专业，用中文。"
        self.messages = [
            {"role":"system", "content": self.system_prompt}
        ]

    def chat(self, user_input, max_turns=5):
        """发送消息并返回回答，自动维护对话历史"""

        self.messages.append({
            "role":"user","content":user_input
        })

        turn_count = 0

        while turn_count < max_turns:

            turn_count += 1

            data = json.dumps({
                "model":"nvidia/nemotron-3.5-lightning:free",
                "messages": self.messages,
                "tools":self.tools,
                "tool_choice":"auto"
            })
            resp = requests.post(url=self.url, headers=self.headers, data=data)
            resp = resp.json()["choices"][0]["message"]

            self.messages.append(resp)

            if not (resp.get("tool_calls")):
                return resp["content"]

            for tc in resp["tool_calls"]:
                func_name = tc["function"]["name"]
                func_args = json.loads(tc["function"]["arguments"])


                if func_name in self.available_functions:
                    func_to_call = self.available_functions[func_name]
                    funciton_result = func_to_call(**func_args)

                    self.messages.append({
                        "role":"tool",
                        "tool_call_id":tc["id"],
                        "content":funciton_result
                    })
        # 超过最大轮数
        self.message.append({
            "role":"user",
            "content":"请基于已有信息给出最终回答。"
        })

        data = json.dumps({
            "model":"nvidia/nemotron-3.5-lightning:free",
            "messages":self.message
        })

        resp2 = requests.post(url=url, headers=headers, data=data)
        return resp2.json()["choices"][0]["message"]["content"]

    def reset(self):
        """重置对话历史"""
        self.message = [
            {"role":"system", "content":self.system_prompt}
        ]

