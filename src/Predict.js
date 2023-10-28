import React, { useState } from "react";
import "./Login.css"
import login from "./images/Screenshot 2023-10-29 at 1.17.32 AM.png"
import predict from "./images/PREDICT.png"
function Predict() {

    const [croptype, setCrop] = useState('')
    const predictBtn = () => {
        console.log(croptype)
    }
    return (
        <div className={"LoginDiv"}>
            <div className={"LoginBox"}>
                <img src={predict} width={150} alt={"Predict"}/>
                <form>
                    <label>Crop Type</label><br/>
                    <input className={"formBox"} type={"text"} value={croptype} onChange={e => setCrop(e.target.value)}/>

                    <br/><br/><br/>


                    <label>Moisture Content</label><br/>
                    <input className={"formBox"} type={"text"}/>
                    <br/><br/><br/>


                    <label>Temperature</label><br/>
                    <input className={"formBox"} type={"text"}/><br/><br/><br/>


                    <button onClick={predictBtn} type={"button"}>Predict</button>
                </form>
            </div>
        </div>
    )
}

export default Predict