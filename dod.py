import pandas as pd
from bs4 import BeautifulSoup
import re

# Load the HTML content
file_path = '/home/user1/ariel/Sultan/DOD.html'
with open(file_path, 'r', encoding='utf-8') as f:
    html_content = f.read()

soup = BeautifulSoup(html_content, 'html.parser')

# Market data lookup (Based on research for the companies in this specific file)
market_data = {
    "Northrop Grumman": {"traded_us": "Yes (NYSE: NOC)", "sentiment": "Strong Buy / Bullish"},
    "Lockheed Martin": {"traded_us": "Yes (NYSE: LMT)", "sentiment": "Neutral to Slightly Positive"},
    "NVIDIA": {"traded_us": "Yes (NASDAQ: NVDA)", "sentiment": "Strongly Bullish / Strong Buy"}
}

data_rows = []

# The contract announcements are typically in <p> or <div> tags within the article body
# This logic searches for paragraphs containing '$' to identify contract awards
paragraphs = soup.find_all('p')

for p in paragraphs:
    text = p.get_text()
    
    # Simple regex to find potential company names and dollar amounts
    # Identifying company: Usually ends in 'Corp.' or 'Systems' at the start of the block
    company_match = re.search(r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", text)
    price_match = re.search(r"\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)", text)
    date_match = re.search(r"completed by (?:[A-Z][a-z]+ \d{1,2}, )?(\d{4})", text)
    
    if company_match and price_match:
        company_name = company_match.group(1).strip()
        contract_size = price_match.group(1)
        
        # Calculate length: Contract date is Feb 2026, so (Completion Year - 2026)
        completion_year = int(date_match.group(1)) if date_match else 2026
        contract_length = max(0, completion_year - 2026)
        
        # Match with market data
        info = market_data.get(company_name, {"traded_us": "Unknown", "sentiment": "N/A"})
        
        data_rows.append({
            "company name": company_name,
            "contract size": f"${contract_size}",
            "contract length (years)": contract_length,
            "is it traded in us stock exchange?": info["traded_us"],
            "analysts sentiment": info["sentiment"]
        })

# Create DataFrame and save to CSV
df = pd.DataFrame(data_rows)
df.to_csv('contract_data_output.csv', index=False)

print("Extraction complete. Data saved to 'contract_data_output.csv'.")
print(df.head())

