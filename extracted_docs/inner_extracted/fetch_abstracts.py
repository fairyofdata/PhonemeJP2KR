import re
import urllib.request
import urllib.error
from bs4 import BeautifulSoup
import json
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

urls = []
with open('output.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('URL: http'):
            urls.append(line.strip().replace('URL: ', ''))

results = []

def fetch_meta(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        html = urllib.request.urlopen(req, timeout=10).read()
        soup = BeautifulSoup(html, 'html.parser')
        
        title = ""
        if soup.title:
            title = soup.title.string.strip()
            
        abstract = ""
        # Try different meta tags for abstract/description
        meta_tags = [
            soup.find('meta', attrs={'name': 'citation_abstract'}),
            soup.find('meta', attrs={'name': 'description'}),
            soup.find('meta', attrs={'property': 'og:description'}),
            soup.find('meta', attrs={'name': 'DC.Description'})
        ]
        for tag in meta_tags:
            if tag and tag.get('content') and len(tag.get('content')) > 20:
                abstract = tag.get('content').strip()
                break
                
        # DBpia specific
        if 'dbpia.co.kr' in url and not abstract:
            abs_div = soup.find('div', class_='abstractTxt')
            if abs_div: abstract = abs_div.text.strip()
            
        return {"url": url, "title": title, "abstract": abstract}
    except Exception as e:
        return {"url": url, "error": str(e)}

academic_urls = [u for u in urls if 'kci.go.kr' in u or 'dbpia.co.kr' in u or 'kisti.re.kr' in u or 'snu.ac.kr' in u or 'hanyang.ac.kr' in u or 'ewha.ac.kr' in u or 'eksss.org' in u]
academic_urls = list(set(academic_urls))

for u in academic_urls:
    res = fetch_meta(u)
    results.append(res)
    print(f"Fetched: {u}")

with open('abstracts.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
