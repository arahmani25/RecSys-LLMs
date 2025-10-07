/**
 * app.js
 * Handles data loading, UI interaction, training loop management,
 * visualization (loss chart, embedding projection), and recommendation generation.
 */

// Global Configuration
const CONFIG = {
    // Model parameters
    maxInteractions: 80000, // Limit memory usage
    embeddingDim: 32,
    batchSize: 512,
    epochs: 20,
    learningRate: 0.001,
    // Visualization/Test parameters
    pcaSample: 1000,
    topK: 10,
    minRatingsForTest: 20,
};

class MovieLensApp {
    constructor() {
        this.interactions = []; // [{userId, itemId, rating, ts, uIdx, iIdx}]
        this.items = new Map(); // itemId -> {title, year, iIdx}
        this.userMap = new Map(); // userId -> uIdx (0-based index)
        this.itemMap = new Map(); // itemId -> iIdx (0-based index)
        this.reverseUserMap = []; // uIdx -> userId
        this.reverseItemMap = []; // iIdx -> itemId
        this.userTopRated = new Map(); // userId -> [{itemId, rating, ts}] (pre-computed historical top-10)
        this.userRatedItems = new Map(); // userId -> Set<itemId> (for exclusion in testing)

        this.model = null;
        this.lossHistory = [];
        this.isTraining = false;
        
        this.initializeUI();
    }
    
    initializeUI() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('train').addEventListener('click', () => this.train());
        document.getElementById('test').addEventListener('click', () => this.test());
        this.updateStatus('Click "Load Data" to start.');
        this.updateTrainButton(false);
        this.updateTestButton(false);
    }

    updateStatus(message) {
        document.getElementById('status').textContent = message;
    }

    updateTrainButton(enabled) {
        document.getElementById('train').disabled = !enabled;
    }

    updateTestButton(enabled) {
        document.getElementById('test').disabled = !enabled;
    }

    /**
     * Data Loading and Preprocessing
     */
    async loadData() {
        this.updateStatus('Loading and parsing data/u.item and data/u.data...');
        this.updateTrainButton(false);
        this.updateTestButton(false);
        
        try {
            // 1. Load u.item (Movie Metadata)
            const itemResponse = await fetch('data/u.item');
            const itemText = await itemResponse.text();
            const itemLines = itemText.trim().split('\n');
            
            let itemIndex = 0;
            itemLines.forEach(line => {
                const parts = line.split('|');
                const itemId = parts[0];
                const title = parts[1];
                
                // Extract year from title, e.g., 'Toy Story (1995)' -> 1995
                const yearMatch = title.match(/\((\d{4})\)/);
                const year = yearMatch ? yearMatch[1] : null;

                if (!this.itemMap.has(itemId)) {
                    this.itemMap.set(itemId, itemIndex);
                    this.reverseItemMap.push(itemId);
                    this.items.set(itemId, { title, year, iIdx: itemIndex });
                    itemIndex++;
                }
            });

            // 2. Load u.data (Interactions)
            const interactionsResponse = await fetch('data/u.data');
            const interactionsText = await interactionsResponse.text();
            const interactionsLines = interactionsText.trim().split('\n');
            
            let userIndex = 0;
            const rawInteractions = [];

            interactionsLines.slice(0, CONFIG.maxInteractions).forEach(line => {
                const [userId, itemId, rating, ts] = line.split('\t');
                
                if (this.itemMap.has(itemId)) {
                    // Create user indexer
                    let uIdx;
                    if (!this.userMap.has(userId)) {
                        uIdx = userIndex;
                        this.userMap.set(userId, userIndex);
                        this.reverseUserMap.push(userId);
                        userIndex++;
                    } else {
                        uIdx = this.userMap.get(userId);
                    }

                    const iIdx = this.itemMap.get(itemId);

                    const interaction = {
                        userId, itemId,
                        rating: parseInt(rating, 10),
                        ts: parseInt(ts, 10),
                        uIdx, iIdx
                    };
                    rawInteractions.push(interaction);

                    // Track rated items
                    if (!this.userRatedItems.has(userId)) {
                        this.userRatedItems.set(userId, new Set());
                    }
                    this.userRatedItems.get(userId).add(itemId);

                    // Track interactions for sorting later
                    if (!this.userTopRated.has(userId)) {
                        this.userTopRated.set(userId, []);
                    }
                    this.userTopRated.get(userId).push(interaction);
                }
            });

            this.interactions = rawInteractions;

            // 3. Compute user's historical top-10
            this.userTopRated.forEach((interactions, userId) => {
                // Sort by rating (desc), then timestamp (desc) for recency tie-breaking
                interactions.sort((a, b) => {
                    if (b.rating !== a.rating) return b.rating - a.rating;
                    return b.ts - a.ts;
                });
                this.userTopRated.set(userId, interactions.slice(0, CONFIG.topK));
            });

            // Find all users who have enough ratings for testing
            this.testUsers = this.reverseUserMap.filter(userId => {
                return this.userRatedItems.get(userId).size >= CONFIG.minRatingsForTest;
            });

            this.numUsers = this.reverseUserMap.length;
            this.numItems = this.reverseItemMap.length;

            this.updateStatus(`Data loaded: ${this.interactions.length} interactions, ${this.numUsers} users, ${this.numItems} items.`);
            this.updateTrainButton(true);

        } catch (error) {
            console.error(error);
            this.updateStatus(`Error loading data. Check console and ensure /data/u.data and /data/u.item exist. Error: ${error.message}`);
        }
    }

    /**
     * Training Pipeline
     */
    async train() {
        if (!this.interactions.length) {
            this.updateStatus('Data not loaded. Click "Load Data" first.');
            return;
        }
        if (this.isTraining) return;

        this.isTraining = true;
        this.updateTrainButton(false);
        this.updateTestButton(false);
        this.lossHistory = [];
        this.clearCanvas('lossChart');
        this.clearCanvas('embeddingChart');

        this.updateStatus(`Initializing model and starting training for ${CONFIG.epochs} epochs...`);

        // Initialize the model
        this.model = new TwoTowerModel(
            this.numUsers,
            this.numItems,
            CONFIG.embeddingDim,
            CONFIG.learningRate
        );

        const interactions = [...this.interactions]; // Shallow copy to shuffle
        const batchSize = CONFIG.batchSize;
        const numBatches = Math.ceil(interactions.length / batchSize);
        
        for (let epoch = 0; epoch < CONFIG.epochs; epoch++) {
            // Shuffle interactions for better training stability
            interactions.sort(() => Math.random() - 0.5);
            let epochLoss = 0;

            for (let i = 0; i < numBatches; i++) {
                const batch = interactions.slice(i * batchSize, (i + 1) * batchSize);
                
                const userIndices = batch.map(d => d.uIdx);
                const itemIndices = batch.map(d => d.iIdx);
                
                const loss = await this.model.trainStep(userIndices, itemIndices);
                epochLoss += loss;
                this.lossHistory.push(loss);

                if (i % 10 === 0 || i === numBatches - 1) {
                    this.updateStatus(`Epoch ${epoch + 1}/${CONFIG.epochs} | Batch ${i + 1}/${numBatches} | Current Loss: ${loss.toFixed(6)}`);
                    this.drawLossChart();
                    await tf.nextFrame(); // Yield for UI updates
                }
            }
        }

        this.isTraining = false;
        this.updateTrainButton(true);
        this.updateTestButton(true);
        this.updateStatus('Training complete. Generating item embedding projection...');
        
        await this.drawEmbeddingProjection();
        tf.disposeVariables(); // Clean up unused tensors
        this.updateStatus('Training complete. Item embedding projection ready. Click "Test" for recommendations.');
    }

    /**
     * Visualization Helpers
     */
    clearCanvas(canvasId) {
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    drawLossChart() {
        const canvas = document.getElementById('lossChart');
        const ctx = canvas.getContext('2d');
        this.clearCanvas('lossChart');

        const { width, height } = canvas;
        const lossValues = this.lossHistory;
        if (lossValues.length === 0) return;

        const maxLoss = Math.max(...lossValues);
        const margin = 20;
        const plotWidth = width - 2 * margin;
        const plotHeight = height - 2 * margin;

        // Draw line plot
        ctx.strokeStyle = '#dc3545';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        
        for (let index = 0; index < lossValues.length; index++) {
            const loss = lossValues[index];
            const x = margin + index * (plotWidth / (lossValues.length - 1));
            const y = height - margin - (loss / maxLoss) * plotHeight;
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
    }

    /**
     * PCA Approximation via SVD for 2D Embedding Projection
     */
    async drawEmbeddingProjection() {
        const canvas = document.getElementById('embeddingChart');
        const ctx = canvas.getContext('2d');
        this.clearCanvas('embeddingChart');
        const { width, height } = canvas;
        const margin = 20;

        await tf.nextFrame();

        const embeddings = this.model.getItemEmbeddings();
        const numItems = embeddings.shape[0];
        
        // Sample indices for faster visualization
        const allIndices = Array.from({length: numItems}, (_, i) => i);
        const sampledIndices = allIndices.sort(() => Math.random() - 0.5).slice(0, CONFIG.pcaSample);
        
        let projected;
        
        tf.tidy(() => {
            const sampledEmbeddings = tf.gather(embeddings, tf.tensor1d(sampledIndices, 'int32'));
            
            // Centering the data
            const mean = sampledEmbeddings.mean(0);
            const centered = sampledEmbeddings.sub(mean);
            
            // SVD: U, S, V^T. V^T contains the principal components (eigenvectors).
            const { V } = tf.linalg.svd(centered);
            
            // Take the first two principal components (2 columns of V)
            const components = V.slice([0, 0], [CONFIG.embeddingDim, 2]); 
            
            // Project the centered data onto the components: X_proj = X_centered @ V_2
            projected = centered.matMul(components).arraySync();
        });

        // Normalize coordinates to fit the canvas
        const xCoords = projected.map(p => p[0]);
        const yCoords = projected.map(p => p[1]);
        
        const minX = Math.min(...xCoords);
        const maxX = Math.max(...xCoords);
        const minY = Math.min(...yCoords);
        const maxY = Math.max(...yCoords);
        
        const rangeX = maxX - minX;
        const rangeY = maxY - minY;

        const scaleX = (width - 2 * margin) / rangeX;
        const scaleY = (height - 2 * margin) / rangeY;

        const points = [];
        ctx.fillStyle = '#007bff';
        ctx.font = '10px Arial';
        
        for (let i = 0; i < projected.length; i++) {
            const originalIndex = sampledIndices[i];
            const itemId = this.reverseItemMap[originalIndex];
            const item = this.items.get(itemId);

            const x = margin + (projected[i][0] - minX) * scaleX;
            const y = height - margin - (projected[i][1] - minY) * scaleY; // Invert y-axis

            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2, true);
            ctx.fill();

            points.push({ x, y, title: item.title });
        }

        // Add hover listener for titles
        canvas.onmousemove = (event) => {
            const rect = canvas.getBoundingClientRect();
            const mouseX = event.clientX - rect.left;
            const mouseY = event.clientY - rect.top;

            let hoverPoint = null;
            for (const point of points) {
                const distSq = (point.x - mouseX) ** 2 + (point.y - mouseY) ** 2;
                if (distSq < 100) { // Check if within 10 pixels radius
                    hoverPoint = point;
                    break;
                }
            }

            // Redraw chart to clear old hover text
            this.clearCanvas('embeddingChart');
            this.drawEmbeddingProjectionPoints(ctx, points);

            if (hoverPoint) {
                ctx.fillStyle = '#333';
                ctx.fillRect(mouseX + 10, mouseY - 20, ctx.measureText(hoverPoint.title).width + 10, 20);
                ctx.fillStyle = 'white';
                ctx.fillText(hoverPoint.title, mouseX + 15, mouseY - 5);
            }
        };
    }

    // Helper to redraw just the points
    drawEmbeddingProjectionPoints(ctx, points) {
        ctx.fillStyle = '#007bff';
        for (const point of points) {
            ctx.beginPath();
            ctx.arc(point.x, point.y, 3, 0, Math.PI * 2, true);
            ctx.fill();
        }
    }


    /**
     * Test Pipeline (Recommendation)
     */
    async test() {
        if (!this.model || this.isTraining) {
            this.updateStatus('Model is not ready. Click "Train" first.');
            return;
        }
        if (this.testUsers.length === 0) {
            this.updateStatus(`No users found with at least ${CONFIG.minRatingsForTest} ratings for testing.`);
            return;
        }

        this.updateStatus('Generating recommendations for a random user...');
        document.getElementById('results').innerHTML = ''; // Clear old results

        // 1. Pick a random user with >= 20 ratings
        const randomUserId = this.testUsers[Math.floor(Math.random() * this.testUsers.length)];
        const userIdx = this.userMap.get(randomUserId);
        
        // 2. Get user's historical top-10 rated movies
        const historicalTop10 = this.userTopRated.get(randomUserId) || [];
        const ratedItemIds = this.userRatedItems.get(randomUserId) || new Set();

        // 3. Compute user embedding and scores against all items
        const userEmbedding = this.model.getUserEmbedding(userIdx);
        const scores = await this.model.getScoresForAllItems(userEmbedding);
        
        // 4. Find top-K recommendations
        const allItemsWithScores = scores.map((score, iIdx) => {
            const itemId = this.reverseItemMap[iIdx];
            return { iIdx, itemId, score };
        });

        // Exclude items the user has already rated
        const unratedItems = allItemsWithScores.filter(item => !ratedItemIds.has(item.itemId));

        // Sort by score (desc) and take top K
        const recommendations = unratedItems
            .sort((a, b) => b.score - a.score)
            .slice(0, CONFIG.topK);

        // 5. Render results
        this.renderResultsTable(randomUserId, historicalTop10, recommendations);

        // Clean up user embedding tensor
        userEmbedding.dispose();
    }

    /**
     * Renders the side-by-side comparison table.
     */
    renderResultsTable(userId, historical, recommendations) {
        const resultsDiv = document.getElementById('results');
        
        let html = `
            <h2>Test Results for User ID: ${userId}</h2>
            <div class="side-by-side">
                <div>
                    <h3>User's Top ${historical.length} Historically Rated Movies</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Movie</th>
                                <th>Rating</th>
                                <th>Year</th>
                            </tr>
                        </thead>
                        <tbody>
        `;
        
        historical.forEach((interaction, index) => {
            const item = this.items.get(interaction.itemId);
            html += `
                <tr>
                    <td>${index + 1}</td>
                    <td>${item.title}</td>
                    <td>${interaction.rating}/5</td>
                    <td>${item.year || 'N/A'}</td>
                </tr>
            `;
        });
        
        html += `
                        </tbody>
                    </table>
                </div>
                <div>
                    <h3>Model's Top ${recommendations.length} Recommended Movies (Unrated)</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Movie</th>
                                <th>Score (Dot Product)</th>
                                <th>Year</th>
                            </tr>
                        </thead>
                        <tbody>
        `;
        
        recommendations.forEach((rec, index) => {
            const item = this.items.get(rec.itemId);
            html += `
                <tr>
                    <td>${index + 1}</td>
                    <td>${item.title}</td>
                    <td>${rec.score.toFixed(4)}</td>
                    <td>${item.year || 'N/A'}</td>
                </tr>
            `;
        });
        
        html += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        
        resultsDiv.innerHTML = html;
        this.updateStatus(`Recommendations generated successfully for User ${userId}.`);
    }
}

// Initialize app when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
});
