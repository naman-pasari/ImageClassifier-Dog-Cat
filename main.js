// main.js
const inputElement = document.getElementById("inputImage");
const previewElement = document.getElementById("previewImage");
const predictButton = document.getElementById("predictButton");
const predictionResult = document.getElementById("predictionResult");

let model;

async function loadModel() {
    model = await tf.loadLayersModel("notebook/models/imageclassifier.h5");
}

loadModel(); // Load the model when the script is loaded

inputElement.addEventListener("change", (event) => {
    const file = event.target.files[0];
    const imageUrl = URL.createObjectURL(file);
    previewElement.src = imageUrl;
});

predictButton.addEventListener("click", async () => {
    if (!model) {
        predictionResult.textContent = "Model not loaded yet. Please wait.";
        return;
    }

    // Preprocess the image
    const image = tf.browser.fromPixels(previewElement);
    const resizedImage = tf.image.resizeBilinear(image, [256, 256]).toFloat();
    const normalizedImage = resizedImage.div(255);

    // Predict the class
    const predictions = model.predict(normalizedImage.reshape([1, 256, 256, 3]));
    const prediction = predictions.arraySync()[0];

    // Display the prediction result
    if (prediction > 0.5) {
        predictionResult.textContent = "Predicted class is Dog";
    } else {
        predictionResult.textContent = "Predicted class is Cat";
    }
});
