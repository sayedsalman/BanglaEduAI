import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import requests
import json

# API endpoint (update if running remotely)
API_URL = "http://localhost:8000/api/v1/analyze"

def analyze(question):
    if not question.strip():
        return "Please enter a question."
    
    try:
        response = requests.post(API_URL, json={"question": question})
        if response.status_code == 200:
            data = response.json()["data"]
            output = "### 📊 Analysis Result\n\n"
            for task, info in data.items():
                label = info["label"]
                conf = info["confidence"] * 100
                output += f"- **{task.replace('_', ' ').title()}**: {label} (Confidence: {conf:.1f}%)\n"
            return output
        else:
            return f"⚠️ API Error: {response.text}"
    except Exception as e:
        return f"⚠️ Could not connect to API. Is the server running? Error: {e}"

# Gradio interface
iface = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(lines=4, placeholder="এখানে আপনার প্রশ্ন লিখুন...", label="Bangla Question"),
    outputs=gr.Markdown(label="Analysis Result"),
    title="🇧🇩 BanglaEduAI — Intelligent Bangla Educational NLP",
    description="Enter a Bangla educational question and get Subject, Topic, Type, Difficulty, and Cognitive Level.",
    examples=[
        ["আলোক সংশ্লেষণ প্রক্রিয়ায় ক্লোরোফিলের ভূমিকা ব্যাখ্যা কর।"],
        ["নিচের কোনটি সালোকসংশ্লেষণের জন্য প্রয়োজনীয় নয়?"],
        ["নিউটনের তৃতীয় সূত্রটি লেখ।"],
    ]
)

if __name__ == "__main__":
    iface.launch()
