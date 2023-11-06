import React, { useState } from "react";
import "./Login.css"
import predict from "./images/PREDICT.png"
import axios from 'axios';
function Predict() {

    const [responseData, setResponseData] = useState(null);

    const [croptype, setCrop] = useState('')
    const [moisture, setMoisture] = useState('')
    const [temp, setTemp] = useState('')

    const formdata = new FormData()
    formdata.append(
        "moisture", moisture
    )
    formdata.append(
        "temperature", temp
    )

    const predictBtn = () => {
        // axios.defaults.headers.common['Content-Type'] = 'application/json'
        // axios.defaults.headers.common['Accept'] = 'application/json'
        //
        axios.post('http://127.0.0.1:8000/api/add_values/', formdata, {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
        }).then(
            response => {
                console.log(response.data)
                setResponseData(response.data);
            }
        ).catch(
            error => {console.log(error)}
        )
        console.log(croptype)
        console.log(moisture)
        console.log(temp)
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
                    <input className={"formBox"} type={"text"} value={moisture} onChange={e => setMoisture(e.target.value)}/>
                    <br/><br/><br/>


                    <label>Temperature</label><br/>
                    <input className={"formBox"} type={"text"} value={temp} onChange={e => setTemp(e.target.value)}/><br/><br/><br/>


                    <button onClick={predictBtn} type={"button"}>Predict</button>
                </form>
                { responseData && (
                    <div className={"result"}>
                        <p>Result: {responseData["pump status"]}</p>
                    </div>

                )}
            </div>
        </div>
    )
}

export default Predict