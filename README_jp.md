[🇺🇸 English](Readme.md) | [🇰🇷 한국어 (Korean)](README_kr.md)

⚠️ *注意：現在進行中のプロジェクト（企画・プロトタイプ段階）です*

[プロジェクト企画 Notion リンク](https://fairydata.notion.site/49b72350b1074fb094e0ec792cff7d59?pvs=4)

# **🗣️ 日本人学習者向け LLMベース音素レベル韓国語発音矯正サービス** 🧑‍🏫📝
[![デモ動画](https://img.youtube.com/vi/4SwwmzEcpZQ/0.jpg)](https://youtu.be/4SwwmzEcpZQ)<br>

> 日本人の韓国語学習者を対象とした、LLMベースの発音矯正ウェブアプリケーションです。
学習者の音声を分析し、「意図した発音」と「実際に認識された発音」を比較します。日本語を母語とする話者特有の発音の癖を分析し、カスタマイズされた矯正フィードバックを提供します。

---

## ✨ 主な機能
- **音声ファイルのアップロードと視覚化**: `wav`, `mp3`, `flac` 形式のオーディオファイルをアップロードし、波形（Waveform）を視覚的に確認できます。
- **意図した発音の認識 (Whisper)**: OpenAIのWhisperモデルを活用し、学習者が意図した韓国語の文章をテキストとして抽出します。
- **実際の発音分析 (Wav2Vec2)**: 韓国語に特化したWav2Vec2モデルを使用し、実際に発声された通りの発音を音素単位で認識します。
- **専門家AIフィードバック (OpenAI GPT-4o-mini)**: 2つの発音の違いを分析し、口の形や舌の位置など、具体的で分かりやすい矯正方法を提案します。

## 🛠 技術スタック
- **Frontend**: Streamlit
- **AI / ML**: 
  - `openai/whisper-small` (意図した発音の認識)
  - `jonatasgrosman/wav2vec2-large-xlsr-53-korean` (実際の発音の認識)
  - `OpenAI API` (GPT-4o-mini によるフィードバック生成)
- **Audio Processing**: Librosa, Matplotlib

## 🚀 インストールと実行方法

1. **リポジトリのクローン**:
   ```bash
   git clone https://github.com/fairyofdata/PhonemeJP2KR
   cd PhonemeJP2KR
   ```

2. **パッケージのインストール**:
   ```bash
   pip install streamlit torch transformers librosa langdetect openai matplotlib
   ```

3. **アプリケーションの実行**:
   ```bash
   cd Main
   streamlit run app.py
   ```

4. **使用方法**:
   - アプリが起動したら、左側のサイドバーに **OpenAI API Key**（`sk-...`）を入力します。
   - 日本人学習者が韓国語で録音した音声ファイルをアップロードします。
   - **"Analyze Pronunciation"** ボタンをクリックし、分析結果とAIからのフィードバックを確認してください！

## ライセンス
本プロジェクトは MIT ライセンスの下で提供されています。