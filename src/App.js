import React, { useState, useEffect, useRef } from "react";
import "./App.css";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";

function App() {
  const className = ["Aku", "Apa", "Apakabar", "Baik", "Kamu", "Nama", "Siapa"];
  const [hasil, setHasil] = useState([]);
  const webcamRef = useRef(null);

  const runCoco = async () => {
    const net = await tf.loadLayersModel(
      "https://cors-anywhere.herokuapp.com/https://virtual-tour.psycopedia.net/model_json/model.json"
    );

    setInterval(() => {
      detect(net);
    }, 500);
  };

  const detect = async (net) => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      const video = webcamRef.current.video;
      // const videoWidth = webcamRef.current.video.videoWidth;
      // const videoHeight = webcamRef.current.video.videoHeight;

      // webcamRef.current.video.width = videoWidth;
      // webcamRef.current.video.height = videoHeight;

      const img = tf.browser.fromPixels(video);
      const resized = tf.image.resizeBilinear(img, [150, 150]);
      const casted = resized.cast("int32");
      const expanded = casted.expandDims(0);
      const obj = net.predict(expanded);
      obj.print();
      // console.log(obj.arraySync()[0]);
      const v = tf.tensor(obj.dataSync());
      const vv = v.argMax().dataSync()[0];
      // console.log(vv);
      // console.log(className[vv]);
      // console.log("SKIP-----------------------------------");
      setHasil([className[vv], obj.arraySync()[0][vv]]);

      tf.dispose(img);
      tf.dispose(resized);
      tf.dispose(casted);
      tf.dispose(expanded);
      tf.dispose(obj);
    }
  };

  useEffect(() => {
    runCoco();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="App">
      <h5>
        {hasil[0]} ({hasil[1]})
      </h5>
      <Webcam
        ref={webcamRef}
        // muted={true}
        audio={false}
        videoConstraints={{ facingMode: "user" }}
        style={{
          position: "absolute",
          marginLeft: "auto",
          marginRight: "auto",
          left: 0,
          right: 0,
          textAlign: "center",
          zindex: 9,
          width: "90%",
        }}
      />
    </div>
  );
}

export default App;
