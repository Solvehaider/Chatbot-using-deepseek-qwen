# Deepseek-R1 and Qwen chatbot using Langchain and ollama


A chatbot application using **DeepSeek-R1** and **Qwen-Max** models, built with **Streamlit** and **LangChain**.

## 🚀 Features
- Supports **DeepSeek-R1** and **Qwen-Max** models
- Upload and process PDF documents
- Retrieves relevant document context for better responses
- Uses **Ollama Embeddings** for in-memory vector storage
- Simple UI with **Streamlit**

## 📌 Installation

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/Solvehaider/Chatbot-using-deepseek-qwen.git
```

### 2️⃣ Create a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

## 🔑 Environment Variables
Create a `.env` file and add your API keys:
```env
QWEN_API_KEY=your_qwen_api_key
GROQ_API_KEY=your_groq_api_key
```

## ▶️ Run the Application
```sh
streamlit run app.py
```

## 📂 Project Structure
```
├── app.py    # Main chatbot application
├── requirements.txt  # Dependencies
├── .env              # API Keys (not committed)
├── README.md         # Documentation
└── /pdfs             # Directory for uploaded PDFs
```

## 🎯 Usage
1. Select a model (**DeepSeek-R1** or **Qwen-Max**)
2. Upload a PDF document
3. Ask questions, and the chatbot will retrieve relevant content from the document

## 🤝 Contributing
Pull requests are welcome! Feel free to open an issue if you find a bug or want to suggest improvements.

## 📜 License
This project is licensed under the License.

---
💡 *Built with love using Streamlit, LangChain, and Alibaba cloud API!* 🚀
