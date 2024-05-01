import requests

url = "http://127.0.0.1:5000/chatbot"

# Example JSON data
data = {
    "symptoms": "Abdominal discomfort, altered bowel habits, with bouts of diarrhea and periods of constipation, including bloating."
}

response = requests.post(url, json=data)

print(response.json())