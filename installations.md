### **Local development setup instructions**

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