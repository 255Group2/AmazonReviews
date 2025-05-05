import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import axios from "axios";

const ProductDetail = () => {
  const { asin } = useParams();
  const navigate = useNavigate(); // for navigating back
  const [reviews, setReviews] = useState([]);
  const [summary, setSummary] = useState("");
  const [accuracy, setAccuracy] = useState("");
  const [page, setPage] = useState(1);

  useEffect(() => {
    fetchReviews(page);
  }, [page, asin]);

  const fetchReviews = (page) => {
    axios
      .get(`http://localhost:8000/reviews/${asin}?page=${page}&limit=10`)
      .then((res) => {
        console.log(res);
        setReviews(res.data.reviews);
        setSummary(res.data.summary);
        setAccuracy(res.data.accuracy);
      })
      .catch((err) => {
        console.error("Error fetching reviews:", err);
      });
  };

  const handlePageChange = (direction) => {
    if (direction === "next") setPage((prev) => prev + 1);
    else if (direction === "prev" && page > 1) setPage((prev) => prev - 1);
  };

  return (
    <>
      <style>
        {`
          .detail-container {
          padding: 30px;
            margin: 0 auto;
            padding: 20px;
            font-family: Arial, sans-serif;
          }

          .summary-box {
            background-color: #fff3cd;
            padding: 16px;
            border: 1px solid #ffeeba;
            border-radius: 5px;
            margin-bottom: 24px;
          }

          .review-card {
            border-left: 6px solid;
            background-color: #f9f9f9;
            padding: 16px;
            border-radius: 6px;
            margin-bottom: 16px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
          }

          .review-positive {
            border-color: #28a745;
            background-color: #e8f5e9;
          }

          .review-negative {
            border-color: #dc3545;
            background-color: #f8d7da;
          }

          .review-ambiguous {
            border-color: #ffc107;
            background-color: #fff3cd;
          }

          .review-title {
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 4px;
          }

          .review-text {
            font-size: 14px;
            margin-bottom: 6px;
          }

          .sentiment-label {
            font-weight: bold;
            font-size: 13px;
          }

          .sentiment-positive {
            color: #155724;
          }

          .sentiment-negative {
            color: #721c24;
          }

          .sentiment-ambiguous {
            color: #856404;
          }

          .meta-info {
            font-size: 12px;
            color: #555;
            margin-top: 4px;
          }

          .verified {
            color: green;
            font-weight: bold;
            margin-left: 10px;
          }

          .pagination {
            text-align: center;
            margin-top: 24px;
          }

          .pagination button {
            padding: 8px 16px;
            margin: 0 6px;
            font-size: 14px;
            background-color: #eee;
            border: 1px solid #ccc;
            border-radius: 4px;
            cursor: pointer;
          }

          .pagination button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
          }
            .review-grid {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
}

.review-card {
  width: 43%;
  margin-bottom: 20px;
}
  .back-button {
            margin-bottom: 20px;
            font-size: 14px;
            background-color: #ddd;
            border: none;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
          }

          .back-button:hover {
            background-color: #ccc;
          }
@media (max-width: 700px) {
  .review-card {
    width: 100%;
  }
}

        `}
      </style>

      <div className="detail-container">
      <button className="back-button" onClick={() => navigate("/")}>
          ← Back to Product List
        </button>
        <div style={{display:"flex",justifyContent:"space-around"}}>
        <h2>Product ID: <strong>{asin}</strong></h2>
        <h2>Accuracy in this document <strong>{accuracy}</strong></h2>
        </div>

        <div className="summary-box">
          <h3>Summary:</h3>
          <p>{summary}</p>
        </div>
        <div className="review-grid">
          {reviews.map((review, index) => {
            const status = review.sentiment_status;

            let cardClass = "";
            if (status === "Misclassified") cardClass = "review-ambiguous";
            else if (status === "Correct" && review.predicted_sentiment === "Positive") cardClass = "review-positive";
            else if (status === "Correct" && review.predicted_sentiment === "Negative") cardClass = "review-negative";
            else cardClass = "review-negative";

            const labelClass =
              status === "Misclassified" ? "sentiment-ambiguous" :
                review.predicted_sentiment === "Positive" ? "sentiment-positive" :
                  "sentiment-negative";

            return (
              <div key={index} className={`review-card ${cardClass}`}>
                <div className="review-title">{review.title}</div>
                <div className="review-text">{review.text}</div>
                <div className={`sentiment-label ${labelClass}`}>
                  Actual: {review.true_label === 1 ? "Positive" : "Negative"} |
                  Predicted: {review.predicted_sentiment} |
                  Status: {status}
                </div>
                <div className="meta-info">
                  By {review.reviewerName} on {review.reviewTime}
                  {review.verified && (
                    <span className="verified">✔ Verified Purchase</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
        <div className="pagination">
          <button onClick={() => handlePageChange("prev")} disabled={page <= 1}>
            Previous
          </button>
          <button onClick={() => handlePageChange("next")}>Next</button>
        </div>
      </div>
    </>
  );
};

export default ProductDetail;
