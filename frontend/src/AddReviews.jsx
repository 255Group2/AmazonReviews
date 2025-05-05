import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const AddReviews = () => {
  const [productId, setProductId] = useState("");
  const [reviews, setReviews] = useState(Array(5).fill(""));
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleReviewChange = (index, value) => {
    const newReviews = [...reviews];
    newReviews[index] = value;
    setReviews(newReviews);
  };

  const addMoreFields = () => {
    setReviews([...reviews, "", "", "", ""]);
  };

  const removeField = (index) => {
    if (reviews.length > 5) {
      const newReviews = reviews.filter((_, i) => i !== index);
      setReviews(newReviews);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Filter out empty reviews
    const validReviews = reviews.filter(review => review.trim() !== "");
    
    if (!productId || validReviews.length === 0) {
      setError("Please enter a product ID and at least one review");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await axios.post("http://localhost:8000/add-reviews", {
        asin: productId,
        reviews: validReviews
      });
      
      setResult(response.data);
    } catch (err) {
      setError("Failed to analyze reviews. Please try again.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment.toLowerCase()) {
      case 'positive':
        return '#4CAF50';
      case 'negative':
        return '#f44336';
      default:
        return '#ff9800';
    }
  };

  return (
    <>
      <style>
        {`
          .add-reviews-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
          }

          .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
          }

          .header h1 {
            font-size: 24px;
            color: #333;
          }

          .back-button {
            padding: 8px 16px;
            background-color: #6c757d;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            text-decoration: none;
          }

          .product-input {
            margin-bottom: 20px;
          }

          .product-input label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
          }

          .product-input input {
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 14px;
          }

          .review-field {
            position: relative;
            margin-bottom: 15px;
          }

          .review-field label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
          }

          .review-textarea {
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 14px;
            min-height: 60px;
            resize: vertical;
          }

          .remove-button {
            position: absolute;
            right: 0;
            top: 25px;
            background-color: #dc3545;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 4px 8px;
            cursor: pointer;
            font-size: 12px;
          }

          .add-fields-button {
            background-color: #6c757d;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            cursor: pointer;
            margin-bottom: 20px;
            font-size: 14px;
          }

          .submit-button {
            width: 100%;
            padding: 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-bottom: 20px;
          }

          .submit-button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
          }

          .error-message {
            color: #dc3545;
            margin-bottom: 20px;
            padding: 10px;
            border: 1px solid #dc3545;
            border-radius: 4px;
            background-color: #f8d7da;
          }

          .results-container {
            margin-top: 40px;
          }

          .summary-container {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 20px;
            margin-bottom: 30px;
          }

          .summary-container h2 {
            margin-bottom: 10px;
            font-size: 20px;
            color: #333;
          }

          .summary-text {
            white-space: pre-wrap;
            line-height: 1.6;
            font-size: 14px;
          }

          .individual-analysis {
            margin-top: 30px;
          }

          .individual-analysis h2 {
            margin-bottom: 20px;
            font-size: 20px;
            color: #333;
          }

          .analysis-card {
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: white;
          }

          .analysis-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
          }

          .review-number {
            font-weight: bold;
            color: #555;
          }

          .sentiment-badge {
            padding: 4px 8px;
            border-radius: 12px;
            color: white;
            font-size: 12px;
            font-weight: bold;
          }

          .review-text {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 10px;
            font-size: 14px;
          }

          .sentiment-details {
            font-size: 12px;
            color: #666;
          }

          .loading {
            text-align: center;
            padding: 20px;
            font-size: 16px;
            color: #666;
          }

          @media (max-width: 600px) {
            .add-reviews-container {
              padding: 10px;
            }
            
            .header {
              flex-direction: column;
              gap: 10px;
              align-items: flex-start;
            }
          }
        `}
      </style>

      <div className="add-reviews-container">
        <div className="header">
          <h1>{!result?"Add Product Reviews":"Report Summary"}</h1>
          <button 
            className="back-button"
            onClick={() => navigate('/')}
          >
            Back to Products
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          {!result&&(<div className="product-input">
            <label>Product ID (ASIN):</label>
            <input
              type="text"
              value={productId}
              onChange={(e) => setProductId(e.target.value)}
              placeholder="Enter product ID"
              required
            />
          </div>)}

          {error && <div className="error-message">{error}</div>}

          {!result && (reviews.map((review, index) => (
            <div key={index} className="review-field">
              <label>Review {index + 1}:</label>
              <textarea
                className="review-textarea"
                value={review}
                onChange={(e) => handleReviewChange(index, e.target.value)}
                placeholder={`Enter review ${index + 1}`}
              />
              {reviews.length > 5 && (
                <button
                  type="button"
                  className="remove-button"
                  onClick={() => removeField(index)}
                >
                  Remove
                </button>
              )}
            </div>
          )))}

          {!result&&(<button
            type="button"
            className="add-fields-button"
            onClick={addMoreFields}
          >
            Add More Review Fields
          </button>)}

          {!result&&(<button
            type="submit"
            className="submit-button"
            disabled={loading}
          >
            {loading ? "Analyzing Reviews..." : "Analyze Reviews"}
          </button>)}
        </form>

        {loading && <div className="loading">Analyzing reviews...</div>}

        {result && (
          <div className="results-container">
            <div className="summary-container">
              <h2>Overall Summary:</h2>
              <p className="summary-text">{result.summary}</p>
            </div>

            <div className="individual-analysis">
              <h2>Individual Review Analysis:</h2>
              {result.individual_analysis.map((analysis, index) => (
                <div key={index} className="analysis-card">
                  <div className="analysis-header">
                    <span className="review-number">Review {index + 1}</span>
                    <span 
                      className="sentiment-badge" 
                      style={{ backgroundColor: getSentimentColor(analysis.predicted_sentiment) }}
                    >
                      {analysis.predicted_sentiment}
                    </span>
                  </div>
                  <div className="review-text">{analysis.text}</div>
                  <div className="sentiment-details">
                    Confidence Score: {(analysis.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </>
  );
};

export default AddReviews;