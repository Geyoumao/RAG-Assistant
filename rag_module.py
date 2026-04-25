from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def load_and_split(file_path, chunk_size=100, chunk_overlap=20):
    try:
        if file_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
        document = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(document)
        return chunks
    except Exception as e:
        print(f"加载失败: {e}")
        return []


def build_vectorstore(chunks, model_name="BAAI/bge-m3"):
    if not chunks:
        raise ValueError("chunks 为空")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def ask_answer(question, vectorstore, k=15, model="deepseek-v4-pro"):
    try:
        results = vectorstore.similarity_search_with_score(question, k=k)
        context = "\n".join(doc.page_content for doc, _ in results)
        context = context[:5000]

        prompt = f"""
        参考以下信息回答问题。如果信息不足，就说「未找到」。请严格按照提供的信息回答，没有出现的不要编造，不要篡改。
        参考信息:{context}
        问题:{question}
        回答:
         """
        API_KEY = os.environ.get("API_KEY")
        if not API_KEY:
            raise ValueError("请在 .env 中设置 API_KEY")
        client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个乐于帮助的留学助手"},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        return f"暂时无法回答问题，请稍后重试。[错误：{e}]"
