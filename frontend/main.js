// main.js
const inputElement = document.getElementById("inputImage");
const previewElement = document.getElementById("previewImage");
const predictButton = document.getElementById("predictButton");
const predictionResult = document.getElementById("predictionResult");

inputElement.addEventListener("change", (event) => {
    const file = event.target.files[0];
    const imageUrl = URL.createObjectURL(file);
    previewElement.src = imageUrl;
});

predictButton.addEventListener("click", async () => {
    const model = await tf.loadLayersModel("model.json");
    const image = tf.browser.fromPixels(previewElement);
    // Preprocess and resize the image if needed
    const predictions = model.predict(image);
    const result = predictions.arraySync()[0]; // Interpret the prediction
    predictionResult.textContent = `Predicted class: ${result}`;
});
