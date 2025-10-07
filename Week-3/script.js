// ------------------------------
// MATRIX FACTORIZATION MODEL
// ------------------------------
let model;

// Initialize when window loads
window.onload = async () => {
  const resultDiv = document.getElementById("result");
  resultDiv.textContent = "Loading data...";

  try {
    await loadData();
    populateDropdowns();
    resultDiv.textContent = "Training model...";

    await trainModel();
    resultDiv.textContent = "Model training completed successfully ✅";
  } catch (err) {
    resultDiv.textContent = "Error loading data or training model: " + err.message;
  }
};

// ------------------------------
// Populate dropdowns for UI
// ------------------------------
function populateDropdowns() {
  const userSelect = document.getElementById("user-select");
  const movieSelect = document.getElementById("movie-select");

  for (let i = 0; i < numUsers; i++) {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `User ${i + 1}`;
    userSelect.appendChild(opt);
  }

  movies.slice(0, 100).forEach(movie => {
    const opt = document.createElement("option");
    opt.value = movie.id;
    opt.textContent = movie.title;
    movieSelect.appendChild(opt);
  });
}

// ------------------------------
// Create matrix factorization model
// ------------------------------
function createModel(numUsers, numMovies, latentDim = 8) {
  // User and movie input layers
  const userInput = tf.input({ shape: [1], name: 'user' });
  const movieInput = tf.input({ shape: [1], name: 'movie' });

  // Embedding layers to learn user/movie latent factors
  const userEmbedding = tf.layers.embedding({
    inputDim: numUsers,
    outputDim: latentDim,
    embeddingsInitializer: 'heNormal'
  }).apply(userInput);

  const movieEmbedding = tf.layers.embedding({
    inputDim: numMovies,
    outputDim: latentDim,
    embeddingsInitializer: 'heNormal'
  }).apply(movieInput);

  // Flatten embedding outputs (from [batch, 1, latentDim] → [batch, latentDim])
  const userVec = tf.layers.flatten().apply(userEmbedding);
  const movieVec = tf.layers.flatten().apply(movieEmbedding);

  // Dot product of user and movie latent vectors → predicted rating
  const dot = tf.layers.dot({ axes: 1 }).apply([userVec, movieVec]);

  // Output layer (optional bias)
  const output = tf.layers.dense({ units: 1 }).apply(dot);

  // Build and return model
  return tf.model({ inputs: [userInput, movieInput], outputs: output });
}

// ------------------------------
// Train the model
// ------------------------------
async function trainModel() {
  model = createModel(numUsers, numMovies);

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError'
  });

  // Prepare data tensors
  const userTensor = tf.tensor1d(ratings.map(r => r.userId), 'int32');
  const movieTensor = tf.tensor1d(ratings.map(r => r.movieId), 'int32');
  const ratingTensor = tf.tensor2d(ratings.map(r => [r.rating]));

  // Train model for few epochs
  await model.fit([userTensor, movieTensor], ratingTensor, {
    batchSize: 64,
    epochs: 5,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) =>
        document.getElementById("result").textContent =
          `Training... epoch ${epoch + 1}, loss=${logs.loss.toFixed(4)}`
    }
  });

  tf.dispose([userTensor, movieTensor, ratingTensor]);
}

// ------------------------------
// Predict rating for selected user/movie
// ------------------------------
async function predictRating() {
  const userId = parseInt(document.getElementById("user-select").value);
  const movieId = parseInt(document.getElementById("movie-select").value);

  if (!model) {
    document.getElementById("result").textContent = "Model not trained yet.";
    return;
  }

  // Prepare input tensors
  const userTensor = tf.tensor2d([[userId]]);
  const movieTensor = tf.tensor2d([[movieId]]);

  // Predict rating
  const pred = model.predict([userTensor, movieTensor]);
  const rating = (await pred.data())[0];

  const movieTitle = movies.find(m => m.id === movieId)?.title || "Unknown movie";
  document.getElementById("result").innerHTML =
    `Predicted rating for <b>User ${userId + 1}</b> on "<b>${movieTitle}</b>": 
     <b>${rating.toFixed(2)}/5</b>`;

  tf.dispose([userTensor, movieTensor, pred]);
}
