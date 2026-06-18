[🇰🇷 한국어 (Korean)](README_kr.md) | [🇯🇵 日本語 (Japanese)](README_jp.md)

# **LLM-based Phoneme-level Korean Pronunciation Correction for Japanese Native Speakers** 🧑‍🏫📝

[![Demo Video](https://img.youtube.com/vi/4SwwmzEcpZQ/0.jpg)](https://youtu.be/4SwwmzEcpZQ)<br>

> A web application designed to help Japanese native speakers improve their Korean pronunciation. This project uses a **4-way linguistic analysis architecture** powered by Wav2Vec2, Whisper, and Gemini 2.5 Flash. It identifies pronunciation errors at the phoneme level and visualizes L1 interference (Katakana mapping) to provide highly targeted pedagogical feedback.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Example Output](#example-output)

---

### **Overview**
As the number of Japanese learners of Korean continues to rise, accurate pronunciation remains a significant challenge. Adult learners often suffer from "phonological deafness," unknowingly mapping unfamiliar Korean sounds to their native Japanese Mora system (L1 Interference). 

This project solves this problem by offering a **4-way linguistic comparison**:
1. **Target**: The intended sentence.
2. **Intelligibility**: How a native Korean listener perceives the pronunciation (simulated via **Whisper**).
3. **Acoustics**: The raw physical sound produced by the learner (extracted via **Wav2Vec2**).
4. **L1 Interference**: A visual Katakana mapping of the learner's actual pronunciation, helping them "notice" their subconscious habits (e.g., unnecessary vowel insertions, failing to close batchim).

### **Key Features**
- **4-Way Linguistic Analysis**: Simultaneously compares Target, Whisper, Wav2Vec2, and Katakana for deep insights.
- **Explicit Phoneme Display (IPA)**: All Korean texts are paired with their exact International Phonetic Alphabet (IPA) transcriptions.
- **Expert LLM Feedback**: Utilizes Google's **Gemini 2.5 Flash** to analyze the data and generate professional, linguistics-grade coaching feedback in Japanese.
- **Japanese UI Localization**: Fully localized for Japanese end-users.
- **Audio Waveform Visualization**: Visual feedback of the spoken audio.

---

### **Architecture**
- **Frontend**: Streamlit
- **Acoustic Phonetic Model**: `kresnik/wav2vec2-large-xlsr-korean`
- **Native Listening Simulator**: `openai/whisper-small`
- **Linguistic Analysis LLM**: Google Gemini 2.5 Flash (`google-generativeai`)
- **Audio Processing**: Librosa, FFmpeg

---

### **Installation**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/fairyofdata/PhonemeJP2KR
   cd PhonemeJP2KR
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you have `ffmpeg` installed on your system for audio processing).*

3. **Set Up Google Gemini API Key**:
   - Create a file named `vertex_key_new.md` in the root directory.
   - Paste your free Gemini API key (from [Google AI Studio](https://aistudio.google.com/)) into this file.

---

### **Usage**

1. **Run the Application**:
   ```bash
   cd Main
   streamlit run app.py
   ```

2. **Input and Record**:
   - Enter your target Korean sentence in the text box.
   - Use your browser microphone to record your pronunciation or upload an audio file.
   - Click the **"発音を分析する" (Analyze Pronunciation)** button.

---

### **Example Output**

The system outputs a detailed 4-column breakdown and an expert coaching message.

![Experiment Result](screencapture-localhost-8501-2026-06-18-22_08_56.png)

### **License**
This project is licensed under the MIT License.
