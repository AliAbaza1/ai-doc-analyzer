import os
from typing import Optional
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def summarize_text(text: str, model: str = "gpt-3.5-turbo") -> Optional[str]:
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Summarize the following text in a clear, concise way."},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        print(f"❌ Error in summarization: {e}")
        return None

def answer_question(context: str, question: str, model: str = "gpt-3.5-turbo") -> Optional[str]:
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Answer the question based on the context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        print(f"❌ Error in Q&A: {e}")
        return None
