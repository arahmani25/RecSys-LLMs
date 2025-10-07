/* ===========================================================
   script.js
   - Defines, trains, and uses a Matrix Factorization model
   - Built using TensorFlow.js entirely in the browser
   =========================================================== */

let model = null;
const resultEl = document.getElementById("result");
const userSelect = document.getElementById("user-select");
const movieSelect = document.getElementById("movie-select");
const predictBtn = document.getElementById("predict-btn");

window.onload = async function () {
  resultEl.textContent = "Loading data...";
  predictBtn.disabled = true;

  await loadData();
  populateSelectors();

  resultEl.textContent = "Training model... please wait.";
  await trainModel();

  resultEl.textContent = "Model training completed successfully!";
  predictBtn.disabled = false;
};

/**
 * Fill the user and movie dropdowns dynamically
 */
function populateSelectors() {
  for (let i = 1; i <= numUsers; i++) {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `User ${i}`;
    userSelect.appendChild(opt);
  }

  Object.keys(movies).slice(0, 300).forEach(id => {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = movies[id];
    movieSelect.appendChild(opt);
  });
}

/**
 * createModel()
 * Build the Matrix Factorization architecture.
 * Each user and movie is represented by a dense vector (embedding).
 * The predicted rating is computed by the dot product of these vectors.
 */
function createModel(numUsers, numMovies, latentDim = 20) {
  // Input layers for user and movie IDs
  const userInput = tf.input({ shape: [1], dtype: "int32" });
  const movieInput = tf.input({ shape: [1], dtype: "int32" });

  // Embedding layers (convert IDs → latent vectors)
  const userEmbed = tf.layers.embedding({
    inputDim: numUsers,
    outputDim: latentDim,
    embeddingsInitializer: "randomNormal"
  }).apply(userInput);

  const movieEmbed = tf.layers.embedding({
    inputDim: numMovies,
    outputDim: latentDim,
    embeddingsInitializer: "randomNormal"
  }).apply(movieInput);

  // Flatten the embeddings to 1D
  const userVec = tf.layers.flatten().apply(userEmbed);
  const movieVec = tf.layers.flatten().apply(movieEmbed);

  // Compute dot product between user and movie embeddings
  const dot = tf.layers.dot({ axes: -1 }).apply([userVec, movieVec]);

  // Final linear layer produces a predicted rating
  const output = tf.layers.dense({ units: 1, activation: "linear" }).apply(dot);

  const model = tf.model({ inputs: [userInput, movieInput], outputs: output });
  return model;
}

/**
 * trainModel()
 * Train the model using the loaded MovieLens data.
 */
async function trainModel() {
  model = createModel(numUsers, numMovies);
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "meanSquaredError"
  });

  // Prepare training data as tensors (zero-based indices)
  const userTensor = tf.tensor2d(ratings.map(r => [r.userId - 1]), [ratings.length, 1], "int32");
  const movieTensor = tf.tensor2d(ratings.map(r => [r.itemId - 1]), [ratings.length, 1], "int32");
  const ratingTensor = tf.tensor2d(ratings.map(r => [r.rating]), [ratings.length, 1]);

  // Train the model (in-browser training)
  await model.fit([userTensor, movieTensor], ratingTensor, {
    epochs: 5,
    batchSize: 64,
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        resultEl.textContent = `Epoch ${epoch + 1}: Loss = ${logs.loss.toFixed(4)}`;
        await tf.nextFrame(); // let UI update
      }
    }
  });

  // Free memory
  userTensor.dispose();
  movieTensor.dispose();
  ratingTensor.dispose();
}

/**
 * predictRating()
 * Predict a rating for the selected user and movie.
 */
async function predictRating() {
  const userId = parseInt(userSelect.value);
  const movieId = parseInt(movieSelect.value);

  if (isNaN(userId) || isNaN(movieId)) {
    resultEl.textContent = "Please select both user and movie.";
    return;
  }

  const userTensor = tf.tensor2d([[userId - 1]], [1, 1], "int32");
  const movieTensor = tf.tensor2d([[movieId - 1]], [1, 1], "int32");

  const prediction = model.predict([userTensor, movieTensor]);
  const predictedValue = (await prediction.data())[0];
  const rounded = Math.min(5, Math.max(1, predictedValue.toFixed(2)));

  resultEl.innerHTML = `<b>Predicted rating</b> for User ${userId} on "<i>${movies[movieId]}</i>": <strong>${rounded}</strong>/5`;

  userTensor.dispose();
  movieTensor.dispose();
  prediction.dispose();
}
