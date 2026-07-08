import json

with open('C:\\Users\\Baek\\Phomene\\extracted_docs\\inner_extracted\\abstracts.json', 'r', encoding='utf-8') as f:
    abstracts = json.load(f)

md_content = "# Academic References & Bibliography\n\nThis document compiles the key academic papers that theoretically ground Phomene's Dual-ASR architecture and L1 Interference rule-based classifier.\n\n"

for item in abstracts:
    title = item.get("title", "Untitled Document").replace(" - 논문 | DBpia", "").strip()
    url = item.get("url", "")
    abstract = item.get("abstract", "")
    error = item.get("error", "")
    
    if error or not abstract or title == "Loading...":
        continue
        
    md_content += f"### [{title}]({url})\n"
    md_content += f"> {abstract}\n\n"
    md_content += "---\n\n"

with open('C:\\Users\\Baek\\Phomene\\docs\\REFERENCES.md', 'w', encoding='utf-8') as f:
    f.write(md_content)
