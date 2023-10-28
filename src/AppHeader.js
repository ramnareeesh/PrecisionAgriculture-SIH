import React from "react";
import "./AppHeader.css"
import { Link } from "react-router-dom"
import logo from "./images/logo.jpg"
function AppHeader() {
    return (
        <nav className={"Nav"}>
            <div className="logo">
                <img src={logo} width={400} alt={"Logo"}/>
            </div>
            <ul className="nav-links">
                <li>
                    <Link to={"/"}>Home</Link>
                </li>
                <li>
                    <Link to={"/about"}>About</Link>
                </li>
                <li>
                    <Link to={"/predict"}>Predict</Link>
                </li>

            </ul>
        </nav>
    )
}

export default AppHeader