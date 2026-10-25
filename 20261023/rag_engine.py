from dotenv import load_dotenv
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 报错：网络环境无法访问 Hugging Face 的服务器（被墙了）
import chromadb
from chromadb.utils import embedding_functions
import json
import requests

class RAGEngine:
    """RAG 引擎，封装检索和生成"""
    def __init__(self,collection_name="oa_knowledge_base"):
        load_dotenv("../.env")
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json"
        }
        # 初始化chromadb
        self.client = chromadb.PersistentClient(path="../chroma_db")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.client.get_collection(name=collection_name,embedding_function=self.embedding_function)

    def retrieve(self,query,top_k=3):
        """检索相关文档"""
        results = self.collection.query(query_texts=[query],n_results=top_k)
        retrieved = []
        for doc,metadata,distance in zip(results["documents"][0],results["metadatas"][0],results["distances"][0]):
            retrieved.append({
                "title": metadata["title"],
                "content": doc,
                "score": 1 - distance
            })
        return retrieved

    def query(self,user_question,top_k=3):
        """完整的RAG查询"""
        relevant_docs = self.retrieve(user_question,top_k)
        context = "\n\n".join([f"【{doc['title']}】:{doc['content']}" for doc in relevant_docs])
        system_prompt = """你是一个企业知识库助手，请基于以下提供的文档内容回答用户问题。
        要求：
        1.只基于文档内容回答问题，不要编造
        2.若文档中没有相关信息，请如实说“文档中没有找到相关信息”
        3.回答时注明参考来源（文档标题）
        4.用简洁专业的中文来回答"""
        user_prompt = f"""参考文档：{context}
        用户问题：{user_question}
        请基于以上文档回答问题
        """
        data = json.dumps({
            "model":"nvidia/nemotron-3.5-lightning:free",
            "messages":[
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_prompt}
            ],
            "temperature":0.3
        })
        resp = requests.post(url=self.url,headers=self.headers,data=data)
        resp = resp.json()["choices"][0]["message"]["content"]

        return resp, relevant_docs