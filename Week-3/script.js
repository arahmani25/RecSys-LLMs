// script.js
// Model definition, training, and prediction logic using TensorFlow.js

let model = null;

// UI elements
const userSelect = document.getElementById('user-select');
const movieSelect = document.getElementById('movie-select');
const resultEl = document.getElementById('result');
const predictBtn = document.getElementById('predict-btn');

window.onload = async function() {
  try {
    resultEl.textContent = 'Loading dataset...';
    predictBtn.disabled = true;

    // loadData is defined in data.js
    await loadData();

    resultEl.textContent = `Data loaded. ${numUsers} users, ${numMovies} movies, ${ratings.length} ratings. Populating selectors...`;

    populateSelectors();
    resultEl.textContent += '\nInitializing and training model in the browser (this may take a little while)...';
    await trainModel();

  } catch (err) {
    resultEl.textContent = 'Error loading data or training model: ' + (err.message || err);
    console.error(err);
    predictBtn.disabled = true;
  }
};

function populateSelectors() {
  // Fill user select (use a reasonable subset if there are many)
  userSelect.innerHTML = '';
  movieSelect.innerHTML = '';

  // Populate users - show all user ids (they are 1-indexed)
  const maxUsersToShow = Math.min(numUsers, 5000); // safe guard
  for (let uid of userIds) {
    const opt = document.createElement('option');
    opt.value = uid;
    opt.textContent = `User ${uid}`;
    userSelect.appendChild(opt);
  }

  // Populate movies - include title where available
  // We'll show all movies that appear in movieIds
  for (let mid of movieIds) {
    const opt = document.createElement('option');
    opt.value = mid;
    const title = movies[mid] || `Movie ${mid}`;
    opt.textContent = `${mid} — ${title}`;
    movieSelect.appendChild(opt);
  }
}

// createModel(numUsers, numMovies, latentDim)
// Embedding-based matrix factorization with user/movie biases and dot product
function createModel(numUsers, numMovies, latentDim = 32) {
  // Inputs are integer indices (1-indexed in dataset). We'll feed zero-based indices
  const userInput = tf.input({shape: [1], dtype: 'int32', name: 'userInput'});
  const movieInput = tf.input({shape: [1], dtype: 'int32', name: 'movieInput'});

  // Embedding layers
  // inputDim must be number of unique items (we'll pass numUsers and numMovies)
  // The embedding layer in tfjs expects indices in [0,inputDim-1]
  const userEmbedLayer = tf.layers.embedding({
    inputDim: numUsers,
    outputDim: latentDim,
    inputLength: 1,
    embeddingsInitializer: 'randomNormal',
    name: 'userEmbedding'
  });

  const movieEmbedLayer = tf.layers.embedding({
    inputDim: numMovies,
    outputDim: latentDim,
    inputLength: 1,
    embeddingsInitializer: 'randomNormal',
    name: 'movieEmbedding'
  });

  // Bias embeddings (outputDim 1) for users and movies
  const userBiasLayer = tf.layers.embedding({
    inputDim: numUsers,
    outputDim: 1,
    inputLength: 1,
    embeddingsInitializer: 'zeros',
    name: 'userBias'
  });

  const movieBiasLayer = tf.layers.embedding({
    inputDim: numMovies,
    outputDim: 1,
    inputLength: 1,
    embeddingsInitializer: 'zeros',
    name: 'movieBias'
  });

  // Apply embeddings
  const userVec = tf.layers.flatten().apply(userEmbedLayer.apply(userInput));
  const movieVec = tf.layers.flatten().apply(movieEmbedLayer.apply(movieInput));

  const userB = tf.layers.flatten().apply(userBiasLayer.apply(userInput));
  const movieB = tf.layers.flatten().apply(movieBiasLayer.apply(movieInput));

  // Dot product of latent vectors -> shape [batch, 1]
  const dot = tf.layers.dot({axes: -1}).apply([userVec, movieVec]);

  // Combine dot product and biases: pred = dot + userBias + movieBias
  const added = tf.layers.add().apply([dot, userB, movieB]);

  // Optionally, allow a small dense layer (here identity linear)
  const output = tf.layers.dense({units: 1, activation: 'linear', name: 'prediction'}).apply(added);

  const mfModel = tf.model({
    inputs: [userInput, movieInput],
    outputs: output,
    name: 'matrixFactorizationModel'
  });

  return mfModel;
}

// trainModel: prepares tensors and fits the model
async function trainModel() {
  resultEl.textContent = 'Preparing tensors...';
  // Use moderate latent dimension
  const latentDim = 32;

  // Create model
  model = createModel(numUsers, numMovies, latentDim);

  // Compile model
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError'
  });

  // Prepare training tensors
  // The embeddings expect zero-based indices, so subtract 1 from userId and itemId.
  const usersArr = ratings.map(r => [r.userId - 1]);
  const moviesArr = ratings.map(r => [r.itemId - 1]);
  const labelsArr = ratings.map(r => [r.rating]);

  // Convert to tensors
  const userTensor = tf.tensor2d(usersArr, [usersArr.length, 1], 'int32');
  const movieTensor = tf.tensor2d(moviesArr, [moviesArr.length, 1], 'int32');
  const labelTensor = tf.tensor2d(labelsArr, [labelsArr.length, 1], 'float32');

  // Free memory of arrays if needed (browser will GC after)
  resultEl.textContent = `Starting training on ${ratings.length} examples...`;

  // Train for a small number of epochs in-browser; adjust as desired
  const epochs = 8;
  const batchSize = 64;
  // We'll use onEpochEnd to update the UI
  await model.fit(
    [userTensor, movieTensor],
    labelTensor,
    {
      epochs,
      batchSize,
      shuffle: true,
      callbacks: {
        onEpochBegin: async (epoch) => {
          resultEl.textContent = `Training... epoch ${epoch + 1} / ${epochs}`;
          await tf.nextFrame(); // yield to render UI
        },
        onEpochEnd: async (epoch, logs) => {
          const loss = (logs && logs.loss) ? logs.loss.toFixed(4) : 'n/a';
          resultEl.textContent = `Epoch ${epoch + 1}/${epochs} finished — loss: ${loss}`;
          await tf.nextFrame();
        }
      }
    }
  );

  // Cleanup tensors we created (model keeps weights)
  userTensor.dispose();
  movieTensor.dispose();
  labelTensor.dispose();

  resultEl.textContent = 'Training complete. Model is ready — choose a user and movie and click "Predict Rating".';
  predictBtn.disabled = false;
}

// predictRating: called by button click
async function predictRating() {
  if (!model) {
    resultEl.textContent = 'Model not ready yet.';
    return;
  }

  // Read selection (values are dataset IDs 1-indexed)
  const selectedUser = parseInt(userSelect.value, 10);
  const selectedMovie = parseInt(movieSelect.value, 10);

  if (Number.isNaN(selectedUser) || Number.isNaN(selectedMovie)) {
    resultEl.textContent = 'Please select both a user and a movie.';
    return;
  }

  resultEl.textContent = `Predicting rating for User ${selectedUser} → Movie ${selectedMovie} ...`;

  // Build tensors (zero-based indices)
  const userIdx = tf.tensor2d([[selectedUser - 1]], [1,1], 'int32');
  const movieIdx = tf.tensor2d([[selectedMovie - 1]], [1,1], 'int32');

  // Predict
  try {
    const predTensor = model.predict([userIdx, movieIdx]);
    // model.predict might return a tensor or array; ensure tensor
    const predData = await predTensor.data();
    const raw = predData[0];

    // Common rating range in Movielens 100k is 1-5
    // Round to 2 decimal places and clamp to [1,5]
    const clamped = Math.max(1, Math.min(5, raw));
    const rounded = Math.round(clamped * 100) / 100;

    // Show average rating for context (optional)
    const existing = ratings.find(r => r.userId === selectedUser && r.itemId === selectedMovie);
    let existingText = existing ? ` (actual: ${existing.rating})` : '';

    resultEl.textContent = `Predicted rating for User ${selectedUser} → "${movies[selectedMovie] || ('Movie ' + selectedMovie)}": ${rounded}${existingText}`;

    // Clean up
    userIdx.dispose();
    movieIdx.dispose();
    if (Array.isArray(predTensor)) predTensor.forEach(t => t.dispose()); else predTensor.dispose();

  } catch (err) {
    console.error('Prediction error:', err);
    resultEl.textContent = 'Prediction failed: ' + (err.message || err);
  }
}
