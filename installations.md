## **Local development setup instructions**

### Repository Structure

```
├── app.py            # Streamlit application entrypoint, Main script to run
├── agents.py         # Multi‑agent workflow definition
├── vectorstore.py    # Embedding and Qdrant search utilities
├── qdrant_setup.py   # First script to run, to generate and store embeddings
├── requirements.txt  # Python dependencies to install
├── .env              # Has been hidden
├── data              # Source files for knowledge base
   ├── Guide_to_Litigation_India.pdf
   ├── Legal_Compliance_ICAI.pdf
├── extra             # Contains additional codes which were used to experiment with during developement (can be ignored)
├── installations.md
└── README.md     
```

---

### Prerequisites

* Python 3.10 (This version has been used for the development)
* A Gemini API key with embedding and chat access (Get yours [here](https://aistudio.google.com/apikey))
* Qdrant instance (self‑hosted or managed) and API key (Create an account [here](https://cloud.qdrant.io/) and then create a cloud cluster)
* Git, GitHub account and [Streamlit community cloud](https://streamlit.io/cloud) (for hosting)

---

### Setup

1. **Clone the repo**

   ```bash
   git clone https://github.com/your-username/legal-chatbot.git
   cd legal-chatbot
   ```

**Optional but recommended: Setup a virtual environment**

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**

   Fill in your keys in `.env`:

   ```ini
   GEMINI_API_KEY=your_gemini_api_key_here
   QDRANT_URL=https://your-qdrant-host
   QDRANT_API_KEY=your_qdrant_api_key_here
   ```

4. **Run locally**

   ```bash
   streamlit run app.py
   ```

   The app will launch in your browser at `http://localhost:8501`.

---