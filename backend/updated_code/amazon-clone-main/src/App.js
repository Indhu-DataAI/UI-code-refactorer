```javascript
import React from "react";
import { BrowserRouter as Router, Route, Switch } from "react-router-dom";
import Home from "./components/home/Home";
import Offers from "./components/Offers"; // Import the new Offers component
import Checkout from "./components/checkout/Checkout";

function App() {
  return (
    <Router>
      <Switch>
        <Route path="/" exact component={Home} />
        <Route path="/offers" component={Offers} /> {/* Add route for Offers */}
        <Route path="/checkout" component={Checkout} />
      </Switch>
    </Router>
  );
}

export default App;
```