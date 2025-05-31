import streamlit as st
import os
from datetime import datetime

import requests
import wikipedia
import speech_recognition as sr
import PyPDF2

import google.generativeai as genai

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain

from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings


# === API Keys ===
genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", "your-gemini-api-key"))
weather_api = st.secrets.get("WEATHER_API_KEY", "your_weatherapi_key")
news_api = st.secrets.get("NEWS_API_KEY", "your_newsdata_io_key")
exchange_rate_api = st.secrets.get("EXCHANGE_RATE_API_KEY", "your_exchangerate_api_key")

# === Page Setup ===
st.set_page_config(
    page_title="Gemini AI Chatbot",
    layout="wide",
    page_icon="🤖"
)
st.title("🤖 Gemini Multi-Tool Chatbot")

# Initialize chat history state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar task selector
task = st.sidebar.selectbox("Choose a task", [
    "General Chat (Gemini)",
    "Weather Info",
    "News Headlines",
    "Math Solver",
    "Wikipedia Summary",
    "Currency Converter",
    "Voice Input Chat",
    "Document Q&A (PDF)"
])

# Clear chat history button
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history.clear()
    # Removed st.experimental_rerun(), Streamlit auto-reruns on state change

# Display recent chat history in sidebar
if st.session_state.chat_history:
    st.sidebar.subheader("Chat History")
    for role, message in st.session_state.chat_history[-5:]:
        display_msg = f"{role}: {message[:50]}..." if len(message) > 50 else f"{role}: {message}"
        st.sidebar.write(display_msg)

# Gemini chat function (unchanged)
def gemini_chat(prompt, chat_history=None):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        if chat_history:
            messages = [
                {"role": "user" if role == "user" else "model", "parts": [{"text": content}]}
                for role, content in chat_history
            ]
            chat = model.start_chat(history=messages)
            response = chat.send_message(prompt)
        else:
            response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# General Chat UI
if task == "General Chat (Gemini)":
    st.subheader("💬 Chat with Gemini AI")

    # Initialize chat session
    if "chat_session" not in st.session_state:
        try:
            st.session_state.chat_session = genai.GenerativeModel("gemini-1.5-flash").start_chat(history=[])
        except Exception as e:
            st.error(f"Failed to initialize chat session: {str(e)}")
            st.stop()

    # Display chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(message.role):
            st.markdown(message.parts[0].text)

    # ⛔ MOVE FOOTERS / CREDITS ABOVE CHAT INPUT
    st.divider()
    st.caption("🤖 Powered by Gemini | Created by You | All rights reserved © 2025")  # FOOTER-LIKE CONTENT

    # ✅ Ensure chat_input is LAST
    prompt = st.chat_input("Ask anything:")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chat_session.send_message(prompt)
                with st.chat_message("assistant"):
                    st.markdown(response.text)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


# === Weather Info ===
elif task == "Weather Info":
    st.subheader("🌦️ Weather Information")
    col1, col2 = st.columns(2)

    with col1:
        city = st.text_input("Enter city:", key="weather_city")
        unit = st.radio("Temperature unit:", ("Celsius", "Fahrenheit"), index=0)

    if city:
        with st.spinner("Fetching weather data..."):
            try:
                url = f"http://api.weatherapi.com/v1/current.json?key={weather_api}&q={city}&aqi=yes"
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                weather_data = r.json()

                if "current" in weather_data:
                    current = weather_data["current"]
                    location = weather_data["location"]

                    with col2:
                        temp = current["temp_c"] if unit == "Celsius" else current["temp_f"]
                        feels_like = current["feelslike_c"] if unit == "Celsius" else current["feelslike_f"]

                        st.metric("Temperature", f"{temp}°{'C' if unit == 'Celsius' else 'F'}")
                        st.write(f"🌡️ **Feels like:** {feels_like}°{'C' if unit == 'Celsius' else 'F'}")
                        st.write(f"📍 **Location:** {location['name']}, {location['country']}")
                        st.write(f"☁️ **Weather:** {current['condition']['text']}")
                        st.write(f"💧 **Humidity:** {current['humidity']}%")
                        st.write(f"🌬️ **Wind Speed:** {current['wind_kph']} km/h ({current['wind_dir']})")
                        st.write(f"👁️ **Visibility:** {current['vis_km']} km")
                        st.write(f"☀️ **UV Index:** {current['uv']}")

                        # Air Quality (optional)
                        aqi = current.get("air_quality")
                        if aqi:
                            st.write(f"🌫️ **Air Quality (US EPA):** {aqi.get('us-epa-index', 'N/A')}")

                    # Weather icon
                    icon_url = current['condition'].get('icon', '')
                    if icon_url and not icon_url.startswith('http'):
                        icon_url = f"https:{icon_url}"
                    if icon_url:
                        st.image(icon_url, width=100)

                    # Last updated time
                    st.caption(f"Last updated: {current.get('last_updated', 'N/A')}")
                else:
                    st.error("City not found or weather data unavailable.")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to fetch weather data: {str(e)}")
            except KeyError as e:
                st.error(f"Unexpected data format: {str(e)}")


# === News Headlines ===
elif task == "News Headlines":
    st.subheader("📰 Latest News Headlines")

    col1, col2 = st.columns(2)
    with col1:
        country = st.selectbox(
            "Select Country",
            [
                ("us", "United States"),
                ("in", "India"),
                ("gb", "United Kingdom"),
                ("au", "Australia"),
                ("ca", "Canada"),
                ("jp", "Japan"),
                ("de", "Germany"),
                ("fr", "France"),
                ("it", "Italy"),
                ("br", "Brazil"),
            ],
            index=0,
            format_func=lambda x: x[1],
        )
    with col2:
        category = st.selectbox(
            "Select Category",
            [
                "top",
                "business",
                "technology",
                "science",
                "sports",
                "health",
                "entertainment",
                "politics",
                "world",
                "environment",
            ],
        )

    col3, col4 = st.columns(2)
    with col3:
        language = st.selectbox(
            "Language",
            [
                ("en", "English"),
                ("es", "Spanish"),
                ("fr", "French"),
                ("de", "German"),
                ("it", "Italian"),
                ("pt", "Portuguese"),
                ("hi", "Hindi"),
                ("ja", "Japanese"),
            ],
            index=0,
            format_func=lambda x: x[1],
        )

    with col4:
        size = st.slider("Number of articles", min_value=1, max_value=10, value=3, step=1)

    if st.button("Get Headlines"):
        with st.spinner("Fetching news headlines..."):
            try:
                url = "https://newsdata.io/api/1/latest"
                params = {
                    "apikey": news_api,
                    "country": country[0],  # country code
                    "language": language[0],  # language code
                    "size": size,
                }
                if category != "top":
                    params["category"] = category

                r = requests.get(url, params=params, timeout=15)
                r.raise_for_status()
                news_data = r.json()

                if news_data.get("status") == "success" and news_data.get("results"):
                    articles = news_data["results"]
                    st.success(f"Found {len(articles)} articles")

                    for i, article in enumerate(articles, 1):
                        with st.expander(f"{i}. {article.get('title', 'No Title')}"):
                            if img_url := article.get("image_url"):
                                try:
                                    st.image(img_url, width=300)
                                except Exception:
                                    st.write("*Image not available*")

                            col_left, col_right = st.columns(2)

                            with col_left:
                                st.markdown(f"**Source:** {article.get('source_id', 'Unknown')}")
                                st.markdown(f"**Source Name:** {article.get('source_name', 'Unknown')}")
                                st.markdown(f"**Published:** {article.get('pubDate', 'Unknown')}")
                                if article.get("creator"):
                                    authors = (
                                        ", ".join(article["creator"])
                                        if isinstance(article["creator"], list)
                                        else str(article["creator"])
                                    )
                                    st.markdown(f"**Author(s):** {authors}")

                            with col_right:
                                if article.get("country"):
                                    countries = (
                                        ", ".join(article["country"])
                                        if isinstance(article["country"], list)
                                        else str(article["country"])
                                    )
                                    st.markdown(f"**Country:** {countries}")
                                if article.get("category"):
                                    categories = (
                                        ", ".join(article["category"])
                                        if isinstance(article["category"], list)
                                        else str(article["category"])
                                    )
                                    st.markdown(f"**Category:** {categories}")
                                if article.get("keywords"):
                                    keywords = (
                                        ", ".join(article["keywords"][:5])
                                        if isinstance(article["keywords"], list)
                                        else str(article["keywords"])
                                    )
                                    st.markdown(f"**Keywords:** {keywords}")

                            if description := article.get("description"):
                                st.markdown(f"**Description:** {description}")

                            if content := article.get("content"):
                                st.markdown("**Full Article Content:**")
                                st.markdown(content)

                            if link := article.get("link"):
                                st.markdown(f"[🔗 Read original article]({link})")

                            st.markdown("---")

                    if total := news_data.get("totalResults"):
                        st.info(f"Total results available: {total}")

                elif news_data.get("status") == "error":
                    error_info = news_data.get("results", {})
                    if isinstance(error_info, dict):
                        code = error_info.get("code", "Unknown error")
                        msg = error_info.get("message", "No error message provided")
                        st.error(f"API Error ({code}): {msg}")
                    else:
                        st.error(f"API Error: {error_info}")
                else:
                    st.warning("No news articles found for the selected criteria.")

            except requests.exceptions.RequestException as e:
                st.error(f"Failed to fetch news: {str(e)}")
            except KeyError as e:
                st.error(f"Unexpected data format from NewsData.io API: {str(e)}")

    # with st.expander("ℹ️ About NewsData.io API"):
    #     st.markdown(
    #         """
    #         **NewsData.io Features:**
    #         - Real-time news from 75,000+ sources
    #         - 60+ languages supported
    #         - 200+ countries coverage
    #         - Multiple categories and filters
    #         - Full article content (when available)
    #         - Metadata including keywords, authors, etc.

    #         **API Limits:**
    #         - Free tier: 200 requests/day
    #         - Response includes up to 50 articles per request
    #         """
    #     )

# === Math Solver ===
elif task == "Math Solver":
    st.subheader("🧮 Math Solver")

    expr = st.text_input("Enter math expression (e.g., 5+3*2, sin(45), log(100)):")

    def safe_eval(expr):
        allowed_chars = set("0123456789+-*/.() ")
        if all(c in allowed_chars for c in expr):
            return eval(expr)
        return None

    if expr:
        with st.spinner("Calculating..."):
            try:
                expr_lower = expr.lower()
                complex_funcs = ['sin', 'cos', 'tan', 'log', 'sqrt']

                if any(func in expr_lower for func in complex_funcs):
                    prompt = f"Calculate this mathematical expression and return only the numerical result without any additional text: {expr}"
                    result = gemini_chat(prompt)
                else:
                    result = safe_eval(expr)
                    if result is None:
                        prompt = f"Calculate this mathematical expression and return only the numerical result: {expr}"
                        result = gemini_chat(prompt)

                st.success(f"**Result:** {result}")
            except Exception as e:
                st.error(f"Error in calculation: {str(e)}. Please check your expression.")

    # Common examples
    st.markdown("**Examples:**")
    for col, example in zip(st.columns(5), ["2 + 3 * 4", "sqrt(16)", "sin(30)", "log(100)", "2**8"]):
        if col.button(example):
            st.session_state.math_expr = example
            st.rerun()


# === Wikipedia Summary ===
elif task == "Wikipedia Summary":
    st.subheader("📚 Wikipedia Summary")

    topic = st.text_input("Enter topic:", key="wiki_topic")
    if topic:
        with st.spinner("Searching Wikipedia..."):
            try:
                results = wikipedia.search(topic)
                if results:
                    selected_page = st.selectbox("Select the most relevant page:", results) if len(results) > 1 else results[0]
                    page = wikipedia.page(selected_page, auto_suggest=False)

                    st.markdown(f"### {page.title}")
                    st.write(wikipedia.summary(selected_page, sentences=5))

                    if page.images:
                        try:
                            st.image(page.images[0], width=300)
                        except:
                            st.write("*Image not available*")

                    st.markdown(f"[Read more on Wikipedia]({page.url})")
                else:
                    st.warning("No Wikipedia page found for this topic.")
            except wikipedia.DisambiguationError as e:
                st.warning(f"Multiple options found. Be more specific. Options: {', '.join(e.options[:5])}")
            except wikipedia.PageError:
                st.error("Page not found. Try a different search term.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


# === Currency Converter ===
elif task == "Currency Converter":
    st.subheader("💱 Currency Converter")

    st.session_state.setdefault('currency_from', 'USD')
    st.session_state.setdefault('currency_to', 'EUR')

    col1, col2, col3 = st.columns(3)
    with col1:
        amt = st.number_input("Amount", value=1.0, min_value=0.0, step=0.01)
    with col2:
        from_cur = st.text_input("From currency (e.g., USD)", value=st.session_state.currency_from).upper()
    with col3:
        to_cur = st.text_input("To currency (e.g., EUR)", value=st.session_state.currency_to).upper()

    if st.button("Convert"):
        with st.spinner("Getting exchange rate..."):
            try:
                url = f"https://open.er-api.com/v6/latest/{from_cur}"
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                data = r.json()

                if data.get("result") == "success":
                    rate = data["rates"].get(to_cur)
                    if rate:
                        result = amt * rate
                        st.success(f"**{amt} {from_cur} = {round(result, 4)} {to_cur}**")
                        st.write(f"Conversion rate: 1 {from_cur} = {rate} {to_cur}")
                        st.caption(f"Last updated: {data.get('time_last_update_utc', 'Unknown')}")
                    else:
                        st.error(f"Currency code `{to_cur}` not found.")
                else:
                    st.error("Failed to retrieve exchange rate.")
            except Exception as e:
                st.error(f"API request failed: {e}")

    # Popular currency shortcuts
    st.markdown("**Popular Conversions:**")
    for col, (fc, tc) in zip(st.columns(5), [("USD", "EUR"), ("USD", "GBP"), ("EUR", "JPY"), ("GBP", "USD"), ("CAD", "USD")]):
        if col.button(f"{fc} → {tc}"):
            st.session_state.currency_from = fc
            st.session_state.currency_to = tc
            st.rerun()

    # API Guide
    # with st.expander("🔍 API Help & Troubleshooting"):
    #     st.markdown(f"""
    #     **API Used**: [ExchangeRate-API v6](https://www.exchangerate-api.com/)
        
    #     **Setup Guide**:
    #     1. Create an account and get your API key.
    #     2. Add it to your `.streamlit/secrets.toml` like this:
    #        ```toml
    #        EXCHANGE_RATE_API_KEY = "your_actual_api_key"
    #        ```

    #     **Common Errors:**
    #     - ❌ 403: Invalid or missing API key
    #     - ❌ 429: Rate limit exceeded
    #     - ❌ 404: Base currency not supported
    #     """)

    # with st.expander("💡 Currency Code Reference"):
    #     st.markdown("""
    #     **Common Currencies:**
    #     - USD (US Dollar), EUR (Euro), GBP (Pound), JPY (Yen)
    #     - INR (Indian Rupee), CAD (Canadian Dollar), AUD (Australian Dollar)
    #     - CHF (Swiss Franc), CNY (Yuan), SEK (Krona)
    #     """)

# === Voice Input Chat ===
elif task == "Voice Input Chat":
    st.subheader("🎙️ Voice Input Chat")
    st.info("Click the button below and speak into your microphone.")

    if st.button("Start Listening", key="voice_button"):
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.write("Listening... (Speak now)")
                r.adjust_for_ambient_noise(source, duration=1)
                audio = r.listen(source, timeout=7, phrase_time_limit=10)

            with st.spinner("Processing your speech..."):
                try:
                    text = r.recognize_google(audio)
                    st.success(f"**You said:** {text}")

                    response = gemini_chat(text)
                    st.markdown("**Gemini Response:**")
                    st.write(response)

                    st.session_state.chat_history.append(("user (voice)", text))
                    st.session_state.chat_history.append(("assistant", response))

                except sr.UnknownValueError:
                    st.error("Could not understand audio.")
                except sr.RequestError as e:
                    st.error(f"Speech recognition service error: {e}")
        except Exception as e:
            st.error(f"Error in voice input: {str(e)}")
            st.info("Ensure microphone is working and permissions are enabled.")

    st.markdown("""
    **Tips for better voice recognition:**
    - Speak clearly and at a moderate pace  
    - Reduce background noise  
    - Keep the microphone close to your mouth  
    - Avoid long pauses while speaking  
    - Ensure microphone permissions are enabled  
    """)

# === Document Q&A (PDF) ===
elif task == "Document Q&A (PDF)":
    st.subheader("📄 Document Q&A")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf", key="pdf_uploader")

    # Initialize session state
    st.session_state.setdefault("processed_document", None)
    st.session_state.setdefault("current_file", None)

    if uploaded_file is not None:
        file_changed = st.session_state.current_file != uploaded_file.name

        if file_changed:
            st.session_state.processed_document = None
            st.session_state.current_file = uploaded_file.name

        if st.session_state.processed_document is None:
            with st.spinner("Processing document..."):
                try:
                    temp_file_path = f"temp_{uploaded_file.name}"
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    reader = PyPDF2.PdfReader(temp_file_path)
                    raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)

                    os.remove(temp_file_path)

                    if not raw_text.strip():
                        st.error("No text could be extracted from the PDF.")
                        st.stop()

                    splitter = CharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        separator="\n"
                    )
                    texts = splitter.split_text(raw_text)

                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    db = FAISS.from_texts(texts, embeddings)

                    memory = ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True
                    )

                    qa_chain = ConversationalRetrievalChain.from_llm(
                        GoogleGenerativeAI(model="gemini-1.5-flash"),
                        db.as_retriever(),
                        memory=memory
                    )

                    st.session_state.processed_document = qa_chain
                    st.success("Document processed successfully!")
                    st.info(f"Document: {uploaded_file.name} | Pages: {len(reader.pages)} | Characters: {len(raw_text):,}")

                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")

        if st.session_state.processed_document:
            question = st.text_input("Ask a question about the document:", key="doc_question")

            if question:
                with st.spinner("Searching for answer..."):
                    try:
                        result = st.session_state.processed_document({"question": question})
                        answer = result.get("answer", "No answer found.")

                        st.markdown("**Answer:**")
                        st.write(answer)

                        with st.expander("See relevant document sections"):
                            docs = result.get("source_documents", [])
                            if docs:
                                for i, doc in enumerate(docs, 1):
                                    st.markdown(f"**Section {i}:**")
                                    st.write(doc.page_content)
                                    st.markdown("---")
                            else:
                                st.write("No specific sections referenced.")
                    except Exception as e:
                        st.error(f"Error answering question: {str(e)}")

# === Footer ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
    <p>🤖 Gemini Multi-Tool Chatbot | Built with Streamlit & Google Gemini AI</p>
    <p>Features: Chat • Weather • News • Math • Wikipedia • Currency • Voice • PDF Q&A</p>
</div>
""", unsafe_allow_html=True)