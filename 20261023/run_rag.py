from rag_engine import RAGEngine

rag = RAGEngine()
answer, relevant_docs = rag.query("加班怎么申请？")
print(answer)

"""
根据《加班审批流程》，申请加班需按以下步骤操作：
- 在OA系统中提前提交加班申请，并注明加班日期、时长及事由；
- 经部门主管审批后方可生效。
- 紧急情况可在加班后24小时内补交申请。

参考来源：【加班审批流程】
"""