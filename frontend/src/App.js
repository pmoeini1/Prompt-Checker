import * as React from 'react';
import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import axios from 'axios'

function App() {
  const [data, setData] = React.useState("");
  const [output, setOutput] = React.useState("")

  React.useEffect(() => {
    console.log(output)
  }, [output])
  
  return (
    <div className="App">
      <header className="App-header">
        <h1>Enter a prompt, and if it returns the wrong prediction, click the appropriate button</h1>
        <input onChange={(e) => {setData(e.target.value)}} />
        {data && <button onClick={() => {
          axios.post('http://localhost:5000/predict', {input_string: data})
            .then((res) => {
              setOutput(res.data);
            })
            .catch((err) => {console.log(err); alert("Error, please try again later.");})
        }}>Predict</button>}
        {output && <h3>Prediction: {output<0.5? "Not malicious" : "Malicious"}</h3>}
        {output && <button onClick={() => {
          axios.post('http://localhost:5000/falsePositive', {input_string: data})
          .then((res) => {alert("Reporting false positive for: " + data);})
          .catch((err) => {alert("Error, please try again later.");})
        }}>False positive</button>}
        {output && <button onClick={() => {
          axios.post('http://localhost:5000/falseNegative', {input_string: data})
          .then((res) => {alert("Reporting false negative for: " + data);})
          .catch((err) => {alert("Error, please try again later.");})
        }}>False negative</button>}
      </header>
    </div>
  );
}

export default App;
