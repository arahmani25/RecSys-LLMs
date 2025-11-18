// script.js

window.onload = async function() {
    const resultElement = document.getElementById('result');
    try {
        resultElement.textContent = "Initializing AI Pipeline & Parsing CSV...";
        resultElement.className = 'loading';
        
        // This calls the function in data.js which parses and processes the CSV
        await loadData();
        
        populateMoviesDropdown();
        resultElement.textContent = "Data Ready. Select a movie.";
        resultElement.className = 'success';
    } catch (error) {
        console.error('Init error:', error);
        resultElement.textContent = "System Error.";
        resultElement.className = 'error';
    }
};

function populateMoviesDropdown() {
    const selectElement = document.getElementById('movie-select');
    // Sort alphabetically
    const sortedMovies = [...movies].sort((a, b) => a.title.localeCompare(b.title));
    
    sortedMovies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        selectElement.appendChild(option);
    });
}

// MATH: Cosine Similarity (The Core of Stage 4)
function calculateCosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

function getRecommendations() {
    const resultElement = document.getElementById('result');
    const selectElement = document.getElementById('movie-select');
    const selectedId = selectElement.value; 
    
    if (!selectedId) {
        resultElement.textContent = "Please select a movie first.";
        resultElement.className = 'error';
        return;
    }

    // Find the source movie
    const likedMovie = movies.find(movie => movie.id == selectedId);

    if (!likedMovie) {
        resultElement.textContent = "Error: Movie not found.";
        return;
    }

    resultElement.textContent = "Analyzing content vectors...";
    resultElement.className = 'loading';

    // Use timeout to let UI update before heavy math
    setTimeout(() => {
        const candidates = movies.filter(m => m.id != likedMovie.id);

        // Calculate Score
        const scoredMovies = candidates.map(candidate => {
            // Compare the vectors created in data.js
            const score = calculateCosineSimilarity(likedMovie.vector, candidate.vector);
            return { ...candidate, score: score };
        });

        // Sort
        scoredMovies.sort((a, b) => b.score - a.score);

        // Get top 3
        const topRecs = scoredMovies.slice(0, 3);

        if (topRecs.length > 0 && topRecs[0].score > 0) {
            // Create a nice output showing the "Why"
            const recList = topRecs.map(m => {
                const tags = [...m.features.themes, ...m.features.subgenres].join(', ');
                return `<li><strong>${m.title}</strong> <br><small>(Matches: ${tags})</small></li>`;
            }).join('');
            
            resultElement.innerHTML = `Because you liked "${likedMovie.title}", we found similar themes:<ul>${recList}</ul>`;
            resultElement.className = 'success';
        } else {
            resultElement.textContent = `No movies with similar themes found.`;
            resultElement.className = 'error';
        }
    }, 50);
}
