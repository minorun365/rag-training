# 必要なPythonライブラリをインポート
import boto3
import json
import numpy as np
import faiss
from dotenv import load_dotenv

load_dotenv()

# AWS SDK for Pythonで、Bedrock用のAPIクライアントを作成
client = boto3.client("bedrock-runtime")

# 生成AIモデルを設定
llm = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
embedding_model = "cohere.embed-multilingual-v3"

# ========================================
# 社内文書を定義
# ========================================
documents = [
    {
        "id": 0,
        "title": "かぐたん",
        "content": "かぐたんはKAG社のSlackチャットボットです。"
    },
    {
        "id": 1,
        "title": "カグカグ",
        "content": "カグカグはKAG社のゆるキャラです。"
    }
]

# ========================================
# 社内文書をベクトルに変換
# ========================================
print("【ベクトルデータを準備】")

# ドキュメントをベクトル化
embeddings = []
for doc in documents:
    print(f"ベクトルに変換中: {doc['title']}")
    
    # ドキュメントの埋め込みを生成
    response = client.invoke_model(
        modelId=embedding_model,
        body=json.dumps({
            "texts": [doc['content']],
            "input_type": "search_document"
        })
    )
    response_body = json.loads(response["body"].read())
    embedding = np.array(response_body["embeddings"][0])
    embeddings.append(embedding)

print()

# NumPy配列に変換し、ベクトルを正規化
embeddings_array = np.array(embeddings).astype('float32')
faiss.normalize_L2(embeddings_array)

# FAISSインデックスを作成
dimension = embeddings_array.shape[1]  # ベクトルの次元数（Cohere: 1024次元）
index = faiss.IndexFlatIP(dimension)   # 内積（コサイン類似度）でインデックス作成
index.add(embeddings_array)            # ベクトルを追加

# ========================================
# 検索を実行
# ========================================
query = "KAG社のゆるキャラの名前は？"

# クエリをベクトル化
response = client.invoke_model(
    modelId=embedding_model,
    body=json.dumps({
        "texts": [query],
        "input_type": "search_query"
    })
)
response_body = json.loads(response["body"].read())

# NumPy配列に変換し、ベクトルを正規化
query_embedding = np.array(response_body["embeddings"][0]).reshape(1, -1).astype('float32')
faiss.normalize_L2(query_embedding)

# コサイン類似度で検索（上位1件を取得）
similarities, indices = index.search(query_embedding, 1)

retrieved_docs = []
for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
    doc = documents[idx]
    retrieved_docs.append(doc)

context = retrieved_docs[0]['content']

print("【検索結果】")
print("クエリ： ", query)
print("結果： ", retrieved_docs)
print()

# ========================================
# 推論を実行
# ========================================
prompt = f"質問： {query} / コンテキスト： {context}"

print("【プロンプト】")
print(prompt)
print()

# LLMに推論を実行
response = client.converse(
    modelId=llm,
    messages=[{
        "role": "user",
        "content": [{"text": prompt}]
    }]
)

# レスポンスを表示
print("【LLMの回答】")
print(response["output"]["message"]["content"][0]["text"])
