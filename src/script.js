import "./style.css";
import { CLASSES } from "../static/classes";

/**
 * Geeting Image as Input
 */
const image_input = document.querySelector("#image-input");
image_input.addEventListener("change", function () {
  const reader = new FileReader();
  reader.addEventListener("load", () => {
    const uploaded_image = reader.result;
    document
      .querySelector("#display-image")
      .setAttribute("src", uploaded_image);
  });
  reader.readAsDataURL(this.files[0]);
});

/**
 * MobileNet Model
 */
let model;
(async function getModifiedMobilenet() {
  model = await tf.loadLayersModel("models/customTrained_mobileNet/model.json");
})();

/**
 * Main Logic
 */
window.predict = async function () {
  /**
   * Preprocessing
   */
  const preprocessTensor = (image, modelName) => {
    let tensor = tf.browser
      .fromPixels(image)
      .resizeNearestNeighbor([224, 224])
      .toFloat();

    //MobileNet
    if (modelName === "mobileNet") {
      let offset = tf.scalar(127.5);
      return tensor.sub(offset).div(offset).expandDims();
    }

    //VGG16
    else if (modelName === "VGG16") {
      let meanImageNetRGB = tf.tensor1d([123.68, 116.779, 103.939]);
      return tensor.sub(meanImageNetRGB).reverse(2).expandDims();
    } else if (modelName === "customTrained") {
      let meanImageNetRGB = tf.tensor1d([123.68, 116.779, 103.939]);
      return tensor.sub(meanImageNetRGB).reverse(2).expandDims();
    } else {
      throw new Error("Model not supported");
    }
  };

  /**
   * Setting up the image
   */
  let image = document.getElementById("display-image");
  const processedTensor = preprocessTensor(image, "VGG16");

  /**
   * Detection
   */
  let predictions = await model.predict(processedTensor).data();
  let result = Array.from(predictions)
    .map((p, i) => {
      return {
        probability: p,
        className: CLASSES[i],
      };
    })
    .sort((a, b) => {
      return b.probability - a.probability;
    })
    .slice(0, 5);

  console.log(
    `prediction: ${result[0].className}, probability: ${result[0].probability}`
  );

  const bg3 = document.getElementsByClassName("bg3")[0];
  bg3.innerHTML = `prediction: ${result[0].className}, probability: ${result[0].probability}`;
};
