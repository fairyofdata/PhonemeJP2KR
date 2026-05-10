🇰🇷 한국어 (Korean) | 🇯🇵 日本語 (Japanese)

⚠️ *Alert: This is an Ongoing Project (Planning & Prototyping Level)*

Project Prototype Planning

# **LLM-based Phoneme-level Korean Pronunciation Correction for Japanese Native Speakers**🧑‍🏫📝
[!영상 제목](https://youtu.be/4SwwmzEcpZQ)<br>

> A web application designed to help Japanese native speakers improve their Korean pronunciation. By leveraging **Whisper** and **Wav2Vec 2.0**, the system identifies discrepancies between intended and actual pronunciation. It provides targeted corrective feedback using **OpenAI's LLM** tailored for Japanese speakers.

---

## ✨ Features
- **Market Demand-Driven Solution**: Responds to the rapid growth of Korean learners in Japan, providing specialized pronunciation training.
- **Audio Upload & Visualization**: Upload `wav`, `mp3`, or `flac` files and visually inspect the audio waveform.
- **Intended Pronunciation Recognition (Whisper)**: Extracts the Korean sentence the learner intended to say.
- **Actual Pronunciation Analysis (Wav2Vec 2.0)**: Analyzes the phonetic output exactly as it sounds using a Korean-specialized Wav2Vec 2.0 model.
- **Expert AI Feedback (OpenAI GPT-4o-mini)**: Compares the two pronunciations and provides specific, actionable feedback on lip and tongue placement, addressing common mistakes made by Japanese speakers.

## 🛠 Tech Stack
- **Frontend**: Streamlit
- **AI / ML**: 
  - `openai/whisper-small` (Intended pronunciation)
  - `jonatasgrosman/wav2vec2-large-xlsr-53-korean` (Actual pronunciation)
  - `OpenAI API` (GPT-4o-mini for feedback generation)
- **Audio Processing**: Librosa, Matplotlib

----
----
## 🚀 Installation & Usage

1. **Clone the Repository**:
   ```bash
   git clone <https://github.com/fairyofdata/PhonemeJP2KR>
   cd PhonemeJP2KR
   ```

2. **Install Required Packages**:
   ```bash
   pip install streamlit torch transformers librosa langdetect openai matplotlib
   ```

3. **Run the Web Interface**:
   ```bash
   cd Main
   streamlit run app.py
   ```

----
4. **How to Use**:
   - Enter your **OpenAI API Key** (`sk-...`) in the left sidebar.
   - Upload a clear audio file of a Japanese speaker speaking Korean.
   - Click **"Analyze Pronunciation"** to view the intended/actual text, waveform, and AI feedback!

----
----
## 📂 Project Structure
```text
Nantokanaru/
 ├── Main/
 │    ├── app.py             # Streamlit Main Application
 │    └── src/
 │         └── models.py     # AI models & LLM feedback logic
 ├── Readme.md               # English README (Default)
 ├── README_kr.md            # Korean README
 └── README_jp.md            # Japanese README
```
