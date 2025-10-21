# 必要なライブラリをインポート
import boto3, streamlit
from dotenv import load_dotenv

load_dotenv()

# フロントエンドを描画
streamlit.title("おしえて！Bedrock")
kb_id = streamlit.text_input("ナレッジベースIDを入力")
model_arn = streamlit.text_input("モデルIDを入力", "anthropic.claude-3-7-sonnet-20250219-v1:0")
question = streamlit.text_input("質問を入力")
button = streamlit.button("質問する")

# Bedrock APIクライアントを作成
bedrock = boto3.client("bedrock-agent-runtime")

# ボタンが押されたらナレッジベースを呼び出し
if button:
    with streamlit.spinner("回答を生成中…"):
        response = bedrock.retrieve_and_generate(
            input={"text": question},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": kb_id,
                    "modelArn": model_arn
                }
            }
        )

    # 回答を描画
    streamlit.write(response["output"]["text"])
