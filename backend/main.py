from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import sys
import os
from pydantic import BaseModel
from typing import List

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from model_loader import load_sentiment_model, load_summary_model, predict_review

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models as None
best_model = None
vectorizer = None
summary_model = None

# Move model loading to startup event
@app.on_event("startup")
async def startup_event():
    """Load models when the application starts"""
    global best_model, vectorizer, summary_model
    print("Loading models...")
    
    try:
        best_model, vectorizer = load_sentiment_model()
        print("✓ Sentiment model loaded")
    except Exception as e:
        print(f"✗ Error loading sentiment model: {e}")
    
    try:
        summary_model = load_summary_model()
        print("✓ Summary model loaded")
    except Exception as e:
        print(f"✗ Error loading summary model: {e}")

REVIEW_FILE_PATH = "data/reviews.json"

class AddReviewsRequest(BaseModel):
    asin: str
    reviews: List[str]

def safe_json(obj):
    """Ensure no NaN or non-serializable data leaks into response."""
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return 0.0
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(x) for x in obj]
    return obj

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global best_model, vectorizer, summary_model
    return {
        "status": "healthy",
        "models_loaded": {
            "sentiment_model": best_model is not None,
            "summary_model": summary_model is not None
        }
    }

# CORS settings for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Model Loading ===
best_model, vectorizer = load_sentiment_model()
summary_model = load_summary_model()



REVIEW_FILE_PATH = "data/reviews.json"


def safe_json(obj):
    """Ensure no NaN or non-serializable data leaks into response."""
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return 0.0
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(x) for x in obj]
    return obj


@app.get("/products")
def get_paginated_products(page: int = Query(1, ge=1), limit: int = Query(20, le=100)):
    """Returns a list of products (unique ASINs) for the dashboard with pagination."""
    start_index = (page - 1) * limit
    seen_asins = set()
    collected = []

    for chunk in pd.read_json(REVIEW_FILE_PATH, lines=True, chunksize=10000):
        for _, row in chunk.iterrows():
            asin = str(row.get("asin", ""))
            if asin not in seen_asins:
                seen_asins.add(asin)
                collected.append(row)
            if len(collected) >= start_index + limit:
                break
        if len(collected) >= start_index + limit:
            break

    selected_rows = collected[start_index:start_index + limit]
    products = []
    for row in selected_rows:
        products.append(safe_json({
            "asin": str(row.get("asin", "")),
            "avg_rating": float(row.get("overall", 0)),
            "total_reviews": 1,
            "first_review": {
                "summary": row.get("summary", ""),
                "reviewerName": row.get("reviewerName", ""),
                "reviewTime": row.get("reviewTime", ""),
                "verified": bool(row.get("verified", False)),
                "reviewText": row.get("reviewText", ""),
                "style": row.get("style", {}),
            },
        }))
    return products

@app.get("/product-ids")
def get_product_ids(limit: int = Query(20, le=100)):
    """Returns a list of product IDs (ASINs) available in the database"""
    asins = []
    
    for chunk in pd.read_json(REVIEW_FILE_PATH, lines=True, chunksize=10000):
        for _, row in chunk.iterrows():
            asin = str(row.get("asin", ""))
            if asin and asin not in asins:
                asins.append(asin)
                if len(asins) >= limit:
                    return asins
        if len(asins) >= limit:
            break
    
    return asins
@app.post("/add-reviews")
def add_reviews(request: AddReviewsRequest):
    """Generate summary and individual analysis for custom reviews"""
    global summary_model, best_model, vectorizer
    
    if summary_model is None or best_model is None or vectorizer is None:
        return {"error": "Models not loaded"}
    
    # Validate input
    if not request.asin:
        return {"error": "Product ID (ASIN) is required"}
    
    if not request.reviews or all(not review.strip() for review in request.reviews):
        return {"error": "At least one review is required"}
    
    # Clean reviews (remove empty ones)
    valid_reviews = [review.strip() for review in request.reviews if review.strip()]
    
    try:
        # Generate overall summary
        summary_text = summary_model.generate_summary(valid_reviews)
        
        # Individual review analysis
        individual_analysis = []
        for review in valid_reviews:
            # Analyze individual review sentiment
            result, prediction, confidence = predict_review(review, "", best_model, vectorizer)
            
            # Ensure prediction is properly formatted
            prediction_value = float(prediction)
            
            individual_analysis.append({
                "text": review,
                "predicted_sentiment": result,
                "confidence": prediction_value
            })
        
        return {
            "asin": request.asin,
            "summary": summary_text,
            "review_count": len(valid_reviews),
            "individual_analysis": individual_analysis
        }
    except Exception as e:
        return {"error": str(e)}    
@app.get("/reviews/{asin}")
def get_product_reviews(asin: str, limit: int = 10, page: int = 1):
    start = (page - 1) * limit
    end = start + limit
    reviews = []
    all_review_texts = []

    correct_predictions = 0
    total_predictions = 0

    for chunk in pd.read_json(REVIEW_FILE_PATH, lines=True, chunksize=10000):
        filtered = chunk[chunk["asin"] == asin]
        for _, row in filtered.iterrows():
            true_label = 1 if row.get("overall", 3) >= 4 else 0

            title = row.get("summary", "")
            text = row.get("reviewText", "")
            result, prediction, _ = predict_review(text, title, best_model, vectorizer)
            prediction_label = 1 if result == "Positive" else 0

            sentiment_status = (
                "Correct" if true_label == prediction_label
                else "Ambiguous" if true_label == 1 and prediction_label == 0
                else "Misclassified"
            )

            if true_label == prediction_label:
                correct_predictions += 1
            total_predictions += 1

            reviews.append({
                "title": title,
                "text": text,
                "reviewerName": row.get("reviewerName", ""),
                "verified": bool(row.get("verified", False)),
                "reviewTime": row.get("reviewTime", ""),
                "true_label": true_label,
                "predicted_sentiment": result,
                "sentiment_status": sentiment_status
            })

            all_review_texts.append(text)
            if len(reviews) >= end:
                break
        if len(reviews) >= end:
            break

    accuracy = round(100 * correct_predictions / total_predictions, 2) if total_predictions else 0.0
    summary_text = summary_model.generate_summary(all_review_texts)

    return safe_json({
        "reviews": reviews[start:end],
        "summary": summary_text,
        "accuracy": accuracy
    })