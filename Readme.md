⚠️ *Alert: This is Ongoing Project (Planning Level)*

[Project Prototype Planning](https://fairydata.notion.site/49b72350b1074fb094e0ec792cff7d59?pvs=4)

# **LLM-based Phoneme-level Korean Pronunciation Correction for Japanese Native Speakers**🧑‍🏫📝
[![영상 제목](https://img.youtube.com/vi/4SwwmzEcpZQ/0.jpg)](https://youtu.be/4SwwmzEcpZQ)<br>
> A project focused on building a **pronunciation correction system** for Japanese native speakers learning Korean, using Whisper and Wav2Vec 2.0 for analyzing intended and actual pronunciation. The system identifies pronunciation errors and provides targeted feedback to help improve Korean pronunciation effectively.

---

## **Table of Contents**

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Example](#example)
6. [Feedback and Contribution](#feedback-and-contribution)
7. [License](#license)

---

### **Overview**

This project addresses the growing demand for Korean language education among Japanese native speakers. As the number of Japanese learners of Korean continues to rise, accurate pronunciation remains a significant challenge, with limited resources tailored to Japanese speakers. This project combines **Whisper** and **Wav2Vec 2.0** to build an effective pronunciation correction system. Whisper determines the intended pronunciation, while Wav2Vec 2.0 analyzes the actual phonetic output, effectively identifying discrepancies and providing feedback. Additionally, **language detection** is used to handle code-switching between Korean and Japanese, ensuring high accuracy in mixed-language conversations.

### **Features**

- **Market Demand-Driven Solution**: Responds to the rapid growth of Korean learners in Japan, where existing resources often lack focus on pronunciation training.
- **Error Detection**: Detects specific pronunciation errors by comparing intended versus actual phonemes.
- **Targeted Feedback**: Provides corrective suggestions designed for Japanese speakers learning Korean.
- **Language Detection**: Handles code-switching (混合言語), making it compatible with conversations that switch between Korean and Japanese.
- **User-Friendly Interface**: Accessible via a web interface built with Streamlit.

---

### **Installation**

To set up this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/fairyofdata/JP2KR_PhonemeTutor
   cd fairyofdata/JP2KR_PhonemeTutor
   ```

2. **Install Required Packages**:
   - Make sure you have Python installed (version 3.8+ recommended).
   - Install the dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Set Up API Keys**:
   - If using Whisper’s API, make sure to set up and configure your OpenAI API key.
   - Add your API key to your environment variables or a `.env` file.

---

### **Usage**

Once installed, you can use the system as follows:

1. **Running the Web Interface**:
   - Start the Streamlit app for a user-friendly interface.
   ```bash
   streamlit run app.py
   ```

2. **Uploading Audio Files**:
   - Users can upload audio files (e.g., extracted from conversation videos).
   - The system will automatically detect code-switching and separate Korean and Japanese sections.

3. **Receiving Pronunciation Feedback**:
   - After processing, the system displays targeted feedback for incorrect pronunciations, highlighting specific phonemes and syllables that require improvement.

---

### **Example**

Here’s a step-by-step example of how to use the system:

1. **Upload an audio file** where a Japanese speaker is speaking Korean. Make sure the audio is clear for better results.
2. **The system detects intended pronunciation** using Whisper and analyzes actual pronunciation with Wav2Vec 2.0.
3. **View Feedback**: The output will highlight the mispronounced syllables and provide recommendations for improvement.

**日本語説明 (Japanese Explanation)**:
1. **オーディオファイルをアップロードしてください**。日本人話者が韓国語を話している音声です。
2. **意図した発音をWhisperが分析し**、Wav2Vec 2.0が実際の発音を確認します。
3. **フィードバックを受け取る**：間違った発音箇所が強調表示され、改善のためのアドバイスが提供されます。

---

### **Feedback and Contribution**

We welcome contributions from the community! If you would like to suggest improvements, report bugs, or request new features, please submit an issue or a pull request.

### **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
