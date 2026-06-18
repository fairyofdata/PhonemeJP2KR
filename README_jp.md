[🇺🇸 English](Readme.md) | [🇰🇷 한국어 (Korean)](README_kr.md)

# **🗣️ 日本人学習者向け LLMベース音素レベル韓国語発音矯正サービス** 🧑‍🏫📝

[![デモ動画](https://img.youtube.com/vi/4SwwmzEcpZQ/0.jpg)](https://youtu.be/4SwwmzEcpZQ)<br>

> 日本人の韓国語学習者を対象とした、LLMベースの発音矯正ウェブアプリケーションです。
学習者の音声を分析し、Wav2Vec2、Whisper、Gemini 2.5 Flashを活用した**「4角言語学的対照分析」**を行います。日本語を母語とする話者特有の発音の癖（L1干渉）を視覚化し、カスタマイズされた矯正フィードバックを日本語で提供します。

---

## ✨ 主な機能
- **4角言語学的対照分析 (4-Way Analysis)**: 
  - 🎯 **目標文章 (Target)**: 意図した基準となる発音
  - 👥 **ネイティブの体感 (Intelligibility - Whisper)**: 韓国語ネイティブの耳に文脈上どう聞こえるかをシミュレーション
  - 🗣️ **物理的な音声 (Acoustics - Wav2Vec2)**: 舌や唇が作り出した純粋な物理的音声
  - 🇯🇵 **L1 干渉 (Katakana)**: 日本語のモーラ（Mora）拍による錯覚をカタカナで視覚化
- **IPA（国際音声記号）併記**: すべての韓国語テキストに正確なIPAを併記し、直感的な比較をサポートします。
- **専門家レベルのAIフィードバック**: **Gemini 2.5 Flash** モデルが調音的なエラーを捉え、言語学の専門家レベルのコーチングを提供します。
- **UIの完全ローカライズ**: エンドユーザーに合わせてインターフェースおよびフィードバックが100%日本語で提供されます。

## 🛠 技術スタック
- **Frontend**: Streamlit
- **Acoustic Model**: `kresnik/wav2vec2-large-xlsr-korean` (実際の発音の認識)
- **Native Listener Model**: `openai/whisper-small` (ネイティブの体感シミュレーション)
- **Linguistic AI**: Google Gemini 2.5 Flash (`google-generativeai`)
- **Audio Processing**: Librosa, FFmpeg

---

## 🚀 インストールと実行方法

1. **リポジトリのクローン**:
   ```bash
   git clone https://github.com/fairyofdata/PhonemeJP2KR
   cd PhonemeJP2KR
   ```

2. **パッケージのインストール**:
   ```bash
   pip install -r requirements.txt
   ```
   *(音声処理のため、OSに `ffmpeg` がインストールされている必要があります。)*

3. **Gemini API キーの設定**:
   - プロジェクトのルートディレクトリに `vertex_key_new.md` ファイルを作成します。
   - [Google AI Studio](https://aistudio.google.com/) で無料で取得したAPIキーをそのファイルに貼り付けます。

4. **アプリケーションの実行**:
   ```bash
   cd Main
   streamlit run app.py
   ```

---

## 💡 実行画面

学習者が発音した音声を基に、スコア、波形、4角対照結果、および詳細な矯正フィードバックが提供されます。

![実験結果](screencapture-localhost-8501-2026-06-18-22_08_56.png)

## ライセンス
本プロジェクトは MIT ライセンスの下で提供されています。