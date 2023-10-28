import React from "react";
import './App.css';
import AppHeader from "./AppHeader"
import { BrowserRouter as Router, Routes, Route} from "react-router-dom";
import Predict from "./Predict"
import Home from "./Home"
function App() {
  return (
    <div className={"App"}>
        <Router>
            <AppHeader/>
            <Routes>
                <Route exact path={"/predict"} element={<Predict/>}/>
                <Route exact path={"/"} element={<Home/>}/>
                <Route exact path={"/predict"} element={<Predict/>}/>
            </Routes>
        </Router>

    </div>
  );
}

export default App;
