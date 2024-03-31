# Stable Diffusion 画像生成
ChatGPT APIを使用してプロンプトを生成し、Stable Diffusionで画像を生成

## インストール
1. ライブラリをインストール
   ```
   pip install -r requirements.txt
   ```
2. OpenAIのAPIキーを設定

## 叩き方
コマンドラインから以下で実行。`--model_path`引数で任意のモデルのパスを指定。
```
python script.py --model_path "path/to/your/stable-diffusion-model"
```
