"""
融合 Agent 命令行工具
用法：
  python hybrid_cli.py                          # 交互模式
  python hybrid_cli.py "BOM-1001 变更了怎么办"   # 单次查询
"""

import sys
from hybrid_agent import HybridAgent

def interactive_mode():
    """交互式对话"""
    agent = HybridAgent()
    print("🤖 融合 Agent 已启动（输入 exit 退出）")
    print("我可以帮你查制度、查 BOM、查库存\n")

    while True:
        try:
            user_input = input("你：")
            if user_input.lower() in ["exit","quit","退出"]:
                print("\n再见")
                break

            answer = agent.chat(user_input)
            print(f"\n助手：{answer}\n")

        except KeyboardInterrupt:
            print("\n再见")
            break

        except Exception as e:
            print(f"\n出错了：{e}\n")

def single_query(question):
    """单次查询"""
    agent = HybridAgent()
    answer = agent.chat(question)
    print(answer)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 从命令行参数读取问题
        question = " ".join(sys.argv[1:])
        single_query(question)
    else:
        interactive_mode()


# # 交互模式
# python hybrid_cli.py
#
# # 单次查询
# python hybrid_cli.py "BOM-1001 变更影响什么？"
# python hybrid_cli.py "BOM-1002 变更了，受影响的物料库存够吗？"