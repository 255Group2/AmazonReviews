## CMPE 255 Team Project: Amazon Product Reviews – Sentiment Analysis & Summarization

### Dataset
We use the publicly available dataset from Kaggle:  
**Amazon Reviews**  
https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews

---

### How to Run the Project

#### 1. Clone the Repository
```bash
git clone https://github.com/255Group2/AmazonReviews.git
cd AmazonReviews/
```

#### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Run the Backend
Make sure you’ve generated `asin_numreviews_allreview.csv` by running the preprocessing notebook in the `summarizer-preprocess` directory.

Then start the FastAPI backend:
```bash
cd backend/
uvicorn main:app --reload --workers 1
```
- If no model is found, the backend will automatically train a model using the `asin_numreviews_allreview.csv` file.

#### 5. Run the Frontend
```bash
cd frontend/
npm install
npm start
```

#### 6. Run the Model Manually (Optional)
```bash
python 255AmazonConfidenceReviews.py
# or
python 255AmazonReviews.py
```

---

### Contributors
- Richard Pham  
- Shivansh Chhabra  
- Ganesh Nagavenkatasai Mohan Kancherla
