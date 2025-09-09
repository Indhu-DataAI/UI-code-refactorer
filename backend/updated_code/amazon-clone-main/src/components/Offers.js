```javascript
import React from "react";
import productsData from "../data/products";
import "../styles/Offers.css"; // Create this CSS file for styling

function Offers() {
  return (
    <div className="offers">
      <h1>Current Offers and Discounts</h1>
      <div className="offers__container">
        {productsData.map((productRow, index) => (
          <div className="offers__row" key={index}>
            {productRow.map((product) => (
              <div className="offer" key={product.id}>
                <img src={product.image} alt={product.name} />
                <h2>{product.name}</h2>
                <p className="offer__price">
                  <span className="original-price">${product.price}</span>
                  <span className="discounted-price">${product.discountedPrice}</span>
                </p>
                <p className="offer__details">{product.offer}</p>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

export default Offers;
```