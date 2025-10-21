# 必要なPyhtonライブラリをインポート
import boto3

# AWS SDK for Pythonで、Bedrock用のAPIクライアントを作成
client = boto3.client("bedrock-runtime")

# Converse APIで推論を行う
response = client.converse(
    modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    messages=[{
        "role": "user",
        "content": [{"text": "KAGのゆるキャラの名前は？"}]
    }]
)

# APIレスポンスから、生成テキストのみを取り出してプリント
print(response["output"]["message"]["content"][0]["text"])
