import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import ProductList from "./ProductList";
import ProductDetail from "./ProductDetail";
import AddReviews from "./AddReviews";

const App = () => (
  <Router>
    <Routes>
      <Route path="/" element={<ProductList />} />
      <Route path="/product/:asin" element={<ProductDetail />} />
      <Route path="/add-reviews" element={<AddReviews />} />
    </Routes>
  </Router>
);

export default App;