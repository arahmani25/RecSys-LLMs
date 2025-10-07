/**
 * two-tower.js
 * Implements the core Two-Tower Retrieval Model using TensorFlow.js.
 */
class TwoTowerModel {
    /**
     * Why Two-Tower?
     * The architecture separates the user and item representations (towers) into independent sub-models.
     * This allows for efficient retrieval (scoring all items against one user) by pre-calculating and indexing all item embeddings.
     * * @param {number} numUsers Total number of unique users.
     * @param {number} numItems Total number of unique items.
     * @param {number} embeddingDim Dimensionality of the embedding space.
     * @param {number} learningRate Learning rate for the Adam optimizer.
     */
    constructor(numUsers, numItems, embeddingDim, learningRate = 0.001) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embeddingDim = embeddingDim;
        
        // User tower: Simple embedding lookup table
        this.userEmbeddings = tf.variable(
            tf.randomNormal([numUsers, embeddingDim], 0, 0.05), 
            true, 
            'user_embeddings'
        );
        
        // Item tower: Simple embedding lookup table
        this.itemEmbeddings = tf.variable(
            tf.randomNormal([numItems, embeddingDim], 0, 0.05), 
            true, 
            'item_embeddings'
        );
        
        // Adam optimizer for stable training
        this.optimizer = tf.train.adam(learningRate);
    }
    
    // User tower: simple embedding lookup
    userForward(userIndices) {
        // tf.gather is used to select rows (embeddings) by index
        return tf.gather(this.userEmbeddings, userIndices);
    }
    
    // Item tower: simple embedding lookup  
    itemForward(itemIndices) {
        // tf.gather is used to select rows (embeddings) by index
        return tf.gather(this.itemEmbeddings, itemIndices);
    }
    
    /**
     * Scoring function: Dot product (similarity metric)
     * Why Dot Product?
     * The dot product (U . I) measures how aligned the user's preference vector (U) is with the item's features vector (I).
     * It's simple, fast, and commonly used in retrieval systems to estimate relevance.
     */
    score(userEmbeddings, itemEmbeddings) {
        // Compute the dot product between two tensors of shape [batch, dim]
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1);
    }
    
    /**
     * Training step using In-Batch Sampled Softmax Loss.
     * @param {number[]} userIndices 
     * @param {number[]} itemIndices 
     * @returns {Promise<number>} Scalar loss value.
     */
    async trainStep(userIndices, itemIndices) {
        // tf.tidy cleans up intermediate tensors for memory management
        return await tf.tidy(() => {
            const userIdxTensor = tf.tensor1d(userIndices, 'int32');
            const itemIdxTensor = tf.tensor1d(itemIndices, 'int32');
            
            // Gradient tape tracks operations for automatic differentiation
            const loss = this.optimizer.compute(
                () => {
                    const userEmbs = this.userForward(userIdxTensor);
                    const itemEmbs = this.itemForward(itemIdxTensor); // These are the positive item embeddings

                    /**
                     * In-Batch Negative Sampling
                     * Logits = U @ I^T (shape N x N). The diagonal is the positive score (U_i . I+_i).
                     * The off-diagonal is the negative score (U_i . I+_j where j != i), using other items in the batch as negatives.
                     * This turns the retrieval problem into a classification problem: which item in the batch is the true positive?
                     */
                    const logits = tf.matMul(userEmbs, itemEmbs, false, true); // U @ I^T
                    
                    // Labels: 1 for the positive pair (diagonal element), 0 otherwise
                    // The true label index is the diagonal (0, 1, 2, ...)
                    const labels = tf.oneHot(
                        tf.range(0, userIndices.length, 1, 'int32'), 
                        userIndices.length
                    );
                    
                    // Softmax cross entropy loss
                    const lossValue = tf.losses.softmaxCrossEntropy(labels, logits);
                    return lossValue;
                }, 
                [this.userEmbeddings, this.itemEmbeddings] // Tensors to be optimized
            );
            
            // Compute gradients and update embeddings
            this.optimizer.applyGradients(loss.grads);
            
            return loss.value.dataSync()[0]; // Return scalar loss
        });
    }
    
    /**
     * Retrieves a single user embedding.
     * @param {number} userIndex 
     * @returns {tf.Tensor} User embedding (shape [embeddingDim]).
     */
    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            return this.userForward([userIndex]).squeeze();
        });
    }
    
    /**
     * Calculates scores for all items against a single user embedding.
     * @param {tf.Tensor} userEmbedding (shape [embeddingDim])
     * @returns {Promise<Float32Array>} Array of scores for all items.
     */
    async getScoresForAllItems(userEmbedding) {
        return await tf.tidy(() => {
            // Compute dot product (MatMul) between all item embeddings and the user embedding
            // itemEmbeddings: [NumItems, Dim] @ userEmbedding: [Dim, 1]
            const scores = tf.matMul(this.itemEmbeddings, userEmbedding.expandDims(1)).squeeze();
            return scores.dataSync();
        });
    }
    
    /**
     * Gets all item embeddings for visualization.
     * @returns {tf.Tensor} Item embeddings tensor (shape [numItems, embeddingDim]).
     */
    getItemEmbeddings() {
        return this.itemEmbeddings;
    }
}
