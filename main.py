from rag_module import load_and_split,ask_answer,build_vectorstore

def main():
    chunks = load_and_split("tokyouniversity.txt")
    if not chunks:
        print("加载文档失败")
        return
    
    vectorstore = build_vectorstore(chunks)
    
    question = ""
    answer = ask_answer(question, vectorstore, k=5)
    print(answer)

if __name__ == "__main__":
    main()