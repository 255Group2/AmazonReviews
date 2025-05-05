import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const ProductList = () => {
  const [products, setProducts] = useState([]);
  const [productPage, setProductPage] = useState(1);
  const [searchId, setSearchId] = useState("");
  const [availableIds, setAvailableIds] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    fetchProducts(productPage);
    fetchAvailableIds();
  }, [productPage]);

  const fetchProducts = (page) => {
    axios
      .get(`http://localhost:8000/products?page=${page}&limit=8`)
      .then((res) => setProducts(res.data))
      .catch((err) => console.error("Failed to fetch products", err));
  };

  const fetchAvailableIds = () => {
    axios
      .get(`http://localhost:8000/product-ids?limit=20`)
      .then((res) => {
        setAvailableIds(res.data);
        setRecommendations(res.data); // Show first 20 as initial recommendations
      })
      .catch((err) => console.error("Failed to fetch product IDs", err));
  };

  const handleProductPageChange = (direction) => {
    if (direction === "next") setProductPage(productPage + 1);
    else if (direction === "prev" && productPage > 1)
      setProductPage(productPage - 1);
  };

  const handleSearch = (e) => {
    e.preventDefault();
    if (searchId) {
      navigate(`/product/${searchId}`);
    }
  };

  const handleRecommendationClick = (asin) => {
    navigate(`/product/${asin}`);
  };

  return (
    <>
      <style>
        {`
          .search-container {
            padding: 20px;
            background-color: #f5f5f5;
            border-radius: 8px;
            margin: 16px auto;
            max-width: 1200px;
          }

          .search-form {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
          }

          .search-input {
            flex: 1;
            padding: 8px 12px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 14px;
          }

          .search-button {
            padding: 8px 16px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
          }

          .search-button:hover {
            background-color: #45a049;
          }

          .recommendations {
            margin-top: 20px;
          }

          .recommendations h3 {
            margin-bottom: 10px;
            font-size: 18px;
            color: #333;
          }

          .recommendation-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
          }

          .recommendation-item {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            text-align: center;
            background-color: white;
            transition: background-color 0.2s;
          }

          .recommendation-item:hover {
            background-color: #e8e8e8;
          }

          .product-container {
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            padding: 16px;
            max-width: 1200px;
            margin: 0 auto;
          }

          .product-card {
            flex: 1 1 calc(33.333% - 16px);
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 12px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            background-color: #fff;
            cursor: pointer;
            transition: box-shadow 0.2s ease-in-out;
          }

          .product-card:hover {
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
          }

          .product-summary {
            font-weight: bold;
            margin-bottom: 6px;
            font-size: 16px;
          }

          .product-meta {
            font-size: 14px;
            color: #666;
            margin-bottom: 8px;
          }

          .product-text {
            font-size: 14px;
            color: #333;
          }

          .verified {
            margin-top: 6px;
            font-size: 12px;
            color: green;
          }

          .pagination {
            text-align: center;
            margin: 20px 0;
          }

          .pagination button {
            margin: 0 5px;
            padding: 6px 12px;
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
            .header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 16px;
  max-width: 1200px;
  margin: 20px auto;
}

.add-reviews-button {
  padding: 8px 16px;
  background-color: #4C4F50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.add-reviews-button:hover {
  background-color: #4C4F50;
        }

          @media (max-width: 900px) {
            .product-card {
              flex: 1 1 calc(50% - 16px);
            }
            .recommendation-grid {
              grid-template-columns: repeat(4, 1fr);
            }
          }

          @media (max-width: 600px) {
            .product-card {
              flex: 1 1 100%;
            }
            .recommendation-grid {
              grid-template-columns: repeat(3, 1fr);
            }
          }
        `}
      </style>

      <div className="search-container">
      <div className="header">
  <h1>Product Reviews</h1>
  <button 
    className="add-reviews-button"
    onClick={() => navigate('/add-reviews')}
  >
    Test Model
  </button>
</div>
        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            className="search-input"
            placeholder="Enter product ID (ASIN)"
            value={searchId}
            onChange={(e) => setSearchId(e.target.value)}
          />
          <button type="submit" className="search-button">
            Search
          </button>
        </form>

        <div className="recommendations">
          <h3>First 20 Available Product IDs:</h3>
          <div className="recommendation-grid">
            {recommendations.map((asin, index) => (
              <div
                key={asin}
                className="recommendation-item"
                onClick={() => handleRecommendationClick(asin)}
              >
                {`${index + 1}. ${asin}`}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="product-container">
        {products.map((p) => (
          <div
            key={p.asin}
            className="product-card"
            onClick={() => navigate(`/product/${p.asin}`)}
          >
            <div className="product-summary">{p.first_review.summary}</div>
            <div className="product-meta">
             {p.avg_rating.toFixed(1)} / 5 • {p.total_reviews} reviews<br />
              {p.first_review.reviewTime} •{" "}
              {p.first_review.style?.["Format:"] || "Unknown Format"}
            </div>
            <div className="product-text">
              {p.first_review.reviewText.length > 150
                ? p.first_review.reviewText.slice(0, 150) + "..."
                : p.first_review.reviewText}
            </div>
            {p.first_review.verified && (
              <div className="verified"> Verified Purchase</div>
            )}
          </div>
        ))}
      </div>

      <div className="pagination">
        <button
          onClick={() => handleProductPageChange("prev")}
          disabled={productPage <= 1}
        >
          Previous
        </button>
        <button onClick={() => handleProductPageChange("next")}>Next</button>
      </div>
    </>
  );
};

export default ProductList;