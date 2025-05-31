# 🤖 Multi-Tool Chatbot

A versatile Streamlit-based chatbot combining **Google Gemini AI** with a suite of useful tools including real-time weather, news headlines, Wikipedia search, voice input, PDF Q&A, math solving, and currency conversion.

---

## 🌟 Features

| Feature              | Description                                                                   |
|----------------------|-------------------------------------------------------------------------------|
| 💬 Gemini Chat        | Natural conversation with Gemini 1.5 Flash AI                                |
| 🌦️ Weather Info       | Real-time weather updates using **WeatherAPI**                              |
| 📰 News Headlines     | Filtered news by country, category, and language using **NewsData.io**       |
| 🧮 Math Solver        | Solve arithmetic and scientific expressions using Gemini or local engine     |
| 📚 Wikipedia Summary  | Fetch summaries and links for any topic from **Wikipedia**                   |
| 💱 Currency Converter | Convert currencies using live rates from **ExchangeRate-API**                |
| 🎙️ Voice Chat         | Convert your speech into a conversation with the AI                          |
| 📄 PDF Q&A            | Upload a PDF and ask questions about its contents using LangChain + FAISS    |

---

## 🧠 Tech Stack

- Frontend: Streamlit (https://streamlit.io/)
- AI Model: Google Gemini (via google.generativeai)
- Knowledge Retrieval: LangChain + FAISS
- Voice Input: SpeechRecognition + PyAudio
- APIs Used:
  - WeatherAPI (https://www.weatherapi.com/)
  - NewsData.io (https://newsdata.io/)
  - ExchangeRate-API (https://www.exchangerate-api.com/)
- Others: Wikipedia API, PyPDF2 for PDF processing

---

## 🚀 Getting Started

### Clone the repository

git clone https://github.com/Nilay-28/ChatBot.git  
cd ChatBot

### Install dependencies

pip install -r requirements.txt

### Configuration

Create a file `.streamlit/secrets.toml` in the project root with the following content:

GEMINI_API_KEY = "your_gemini_api_key"  
WEATHER_API_KEY = "your_weather_api_key"  
NEWS_API_KEY = "your_newsdata_io_key"  
EXCHANGE_RATE_API_KEY = "your_exchange_rate_api_key"

### Running the App

streamlit run ChatBot.py

Then open http://localhost:8501 in your browser.

---

## ✨ Example Use Cases

- 🧑‍🎓 Students: Solve math problems, extract notes from PDFs, get quick Wikipedia knowledge  
- 👨‍💼 Professionals: Read breaking news, convert currency, use voice input for faster interaction  
- 🌍 General Users: Chat with AI, check weather updates, ask questions about documents

---

## 📞 Contact

Have feedback, suggestions, or questions?

- Your LinkedIn: https://www.linkedin.com/in/nilaykoul2807/
- Open an Issue: https://github.com/Nilay-28/ChatBot/issues

---

✅ Let me know if you want this as a file or a Streamlit Cloud–ready version (with app.py and secrets.toml).

---
