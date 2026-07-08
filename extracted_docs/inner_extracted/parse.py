from bs4 import BeautifulSoup
import io

with open('기획안 한국어 학습 일본어 모어 화자를 위한 오디오 트레이닝 서비스 49b72350b1074fb094e0ec792cff7d59.html', 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f, 'html.parser')

with io.open('output.txt', 'w', encoding='utf-8') as out:
    out.write("=== TEXT EXTRACT ===\n")
    for p in soup.find_all(['h1', 'h2', 'h3', 'p', 'li']):
        out.write(p.text.strip() + "\n")

    out.write("\n=== LINKS ===\n")
    for a in soup.find_all('a', href=True):
        out.write(f"Text: {a.text.strip()}\nURL: {a['href']}\n\n")
