
1. The Movie Brain Got Smarter (Deep Learning) 🧠
Old Way: The model only knew a movie by its simple ID number. It was like calling a movie "Movie #5" and hoping the computer figured out what it was like over time.
This is called a lookup table.

New Way: We gave the model the movie's Genre features (Action, Comedy, Drama, etc.) as the starting point. We then ran these features through a small, 
smart network called an MLP (Multi-Layer Perceptron) or a "deep learning layer."

Result: The Item Tower is no longer a simple table; it's a function that looks at the genres and calculates a high-quality,
descriptive embedding for the movie. This helps the system recommend movies it hasn't seen before.

2. We Added a Benchmark Test for Comparison 📊
We now have two recommendation lists in the final display:

Deep Learning Model: The new, smarter recommendations using Genres and the MLP.

Simple Embedding Baseline: The recommendations from the basic, "old-way" model (just ID numbers).

Reason: This comparison lets us clearly see, side-by-side, if all the effort and complexity of adding Deep Learning and Genre features actually leads to better movie suggestions for the user.


1. two-tower.js (Core Deep Learning Architecture)
The entire structure of the model was changed from a simple variable-based lookup to a tf.layers based Deep Learning network.

Area of Change	Code Changed	Description of Change
Model Type	Removed tf.variable for embeddings.	The model now uses tf.layers.sequential for both towers, which is the standard way to build Deep Learning models in TensorFlow.
User Tower	Changed userEmbeddings = tf.variable(...) to use tf.layers.embedding and tf.layers.flatten.	While still an ID-lookup, it's formalized as a layer structure, making it compatible with the new training pipeline.
Item Tower	Replaced itemEmbeddings = tf.variable(...) with a two-layer MLP: tf.layers.dense({inputShape: [numGenres], ...}) → tf.layers.dense(...).	This is the most significant change. It transforms the Item Tower into a Deep Learning component that processes Genre features instead of simple Item IDs.
Input/Forward Pass	itemForward(itemIndices) changed to itemForward(itemFeatures).	The Item Tower no longer accepts an ID; it accepts the multi-hot Genre vector.
Utility Methods	Added getUserEmbeddingMatrix() and updated getAllItemEmbeddings(allItemFeatures).	These are necessary to extract the learned ID vectors for the Simple Embedding (Baseline) comparison and to predict embeddings for all items using the new MLP.

Export to Sheets
2. app.js (Data Processing and Comparison Logic)
This file handles the new Genre data and the implementation of the three-way comparison test.

Area of Change	Code Changed	Description of Change
Configuration	Added hiddenDim: 64 to the CONFIG object.	This sets the size of the hidden layer for the Item Tower's MLP.
Data Loading	Updated the parsing of data/u.item using parts.slice(5).map(Number).	The code now explicitly extracts the 19 genre flags from the raw movie data and stores them as item.features.
Feature Preparation	New tensor created: this.allItemFeaturesTensor.	This stores all movies' genre vectors, which is used for all item-side predictions after training.
Training Input	The batch creation logic now extracts itemFeaturesBatch from the data instead of itemIndices.	Training feeds the Genre features into the Item Tower's MLP.
Test Pipeline	The testing logic was split into three parts:	This implements your three-section view requirement:
1. DL Recommendation: Calls model.userForward() and uses the MLP-derived item embeddings.	
2. Simple Embedding Rec.: Calls model.getUserEmbeddingMatrix() to get the raw user ID vector and uses it with the same item embeddings.	
3. Historical Data: Retrieves userTopRated.	
Result Rendering	renderResultsTable() was heavily refactored to accept three distinct data sets and wrap the three generated HTML tables within a <div class="side-by-side">.	This ensures the final output is in the required three-column format (Historical

Export to Sheets
3. index.html
Area of Change	Code Changed	Description of Change
Titles	Updated titles to include "Deep Two-Tower" and "(MLP + Genres)".	Reflects the new architecture.
CSS (for Layout)	Added inline styles (style="display: flex; gap: 20px;" and style="flex: 1 1 33%;") to the <div class="side-by-side"> and its children.	This is crucial for the three-column layout to work without external CSS.
Notes Section	Updated the "Usability Tips & Setup" to explain the new MLP architecture and the Baseline Comparison.	



# MovieLens 100K Deep Two-Tower Retrieval Demo (TensorFlow.js)

This project implements a client-side Two-Tower Retrieval model using TensorFlow.js for the MovieLens 100K dataset, suitable for static hosting environments like GitHub Pages.

## Key Architectural Update: Deep Learning Item Tower

The original implementation used a simple embedding lookup table for both the User and Item towers. This revised version introduces Deep Learning to the Item Tower by incorporating rich item features (genres) and processing them through a Multi-Layer Perceptron (MLP).

The overall architecture is:

1.  User Tower (Simple): `user_id` $\rightarrow$ Embedding Lookup $\rightarrow$ `User Embedding`
2.  Item Tower (Deep): `genre_multi_hot_vector` $\rightarrow$ Dense (ReLU) $\rightarrow$ Dense (Linear) $\rightarrow$ `Item Embedding`
3.  Scoring: `User Embedding` $\cdot$ `Item Embedding` (Dot Product)
4.  Loss: In-Batch Sampled Softmax

## Detailed Changes by File

---

### `two-tower.js`

The `TwoTowerModel` class was refactored to use `tf.layers.sequential` for both towers, facilitating the use of Deep Learning layers.

| Component | Original Approach (Simple) | Revised Approach (Deep Learning) |
| :--- | :--- | :--- |
| User Tower | `tf.variable` (Manual Embedding Table) | `tf.layers.embedding` + `tf.layers.flatten` (Still ID-based lookup, but now a formal Layer) |
| Item Tower | `tf.variable` (Manual Embedding Table) | `tf.layers.dense` (ReLU, Hidden Dim) $\rightarrow$ `tf.layers.dense` (Embedding Dim, Output) |
| Training Input | `userIndices`, `itemIndices` (IDs) | `userIndices` (IDs), `itemFeatures` (Genre Vectors) |
| Prediction | `tf.gather` on `itemEmbeddings` | `itemModel.predict` on `allItemFeaturesTensor` |
| Comparison Prep | N/A | Added `getUserEmbeddingMatrix()` to access the user embedding weights for baseline comparison. |

---

### `app.js`

This file underwent extensive changes for data parsing and the new testing pipeline.

| Feature | Original Implementation | Revised Implementation |
| :--- | :--- | :--- |
| Data Parsing | Only parsed `itemId`, `title`, `userId`, `rating`, `ts`. | New: Parses the 19 Genre columns from `u.item` into multi-hot vectors (`item.features`). |
| Feature Tensor | N/A | New: Creates `this.allItemFeaturesTensor` (`[NumItems, NumGenres]`) for fast batch prediction by the MLP. |
| Training Batching | Batched `uIdx` and `iIdx`. | Batches `uIdx` and the corresponding Item Genre Features (used by the Item Tower MLP). |
| Test Pipeline | Generated a single recommendation list. | Enhanced: Now generates two recommendation lists: 1) Deep Learning Rec. (MLP/Genre) and 2) Simple Embedding Rec. (Baseline, using only ID-embeddings). |
| Visualization | `drawEmbeddingProjection()` now retrieves embeddings from `model.getAllItemEmbeddings(this.allItemFeaturesTensor)`, ensuring the visualization reflects the Deep Learning-derived Item Embeddings. |

---

### `index.html`

Minor adjustments were made to the front-end structure and styling.

1.  Layout: Adjusted the `.container` width and added CSS for a new `.side-by-side.three-columns` layout to display the Historical, Deep Learning, and Baseline recommendations simultaneously.
2.  Text: Updated titles and tips to reflect the new MLP + Genres architecture and the comparison feature.
