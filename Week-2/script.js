// Initialize the application when the window loads
window.onload = async function() {
    try {
        // Display loading message
        const resultElement = document.getElementById('result');
        resultElement.textContent = "Loading movie data...";
        resultElement.className = 'loading';
        
        // Load data
        await loadData();
        
        // Populate dropdown and update status
        populateMoviesDropdown();
        resultElement.textContent = "Data loaded. Please select a movie.";
        resultElement.className = 'success';
    } catch (error) {
        console.error('Initialization error:', error);
        // Error message already set in data.js
    }
};

// Populate the movies dropdown with sorted movie titles
function populateMoviesDropdown() {
    const selectElement = document.getElementById('movie-select');
    
    // Clear existing options except the first placeholder
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }
    
    // Sort movies alphabetically by title
    const sortedMovies = [...movies].sort((a, b) => a.title.localeCompare(b.title));
    
    // Add movies to dropdown
    sortedMovies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        selectElement.appendChild(option);
    });
}

// Main recommendation function
function getRecommendations() {
    const resultElement = document.getElementById('result');
    
    try {
        // Step 1: Get user input
        const selectElement = document.getElementById('movie-select');
        const selectedMovieId = parseInt(selectElement.value);
        
        if (isNaN(selectedMovieId)) {
            resultElement.textContent = "Please select a movie first.";
            resultElement.className = 'error';
            return;
        }
        
        // Step 2: Find the liked movie
        const likedMovie = movies.find(movie => movie.id === selectedMovieId);
        if (!likedMovie) {
            resultElement.textContent = "Error: Selected movie not found in database.";
            resultElement.className = 'error';
            return;
        }
        
        // Show loading message while processing
        resultElement.textContent = "Calculating recommendations...";
        resultElement.className = 'loading';
        
        // Use setTimeout to allow the UI to update before heavy computation
        setTimeout(() => {
            try {
                // Step 3: Prepare for similarity calculation
                const likedGenres = new Set(likedMovie.genres);
                const allGenres = new Set(movies.flatMap(movie => movie.genres));
                const genreList = Array.from(allGenres);

                const likedVector = genreList.map(genre => likedGenres.has(genre) ? 1 : 0);
                
                const candidateMovies = movies.filter(movie => movie.id !== likedMovie.id);
                
                // Step 4: Calculate Cosine similarity scores
                const scoredMovies = candidateMovies.map(candidate => {
                    const candidateGenres = new Set(candidate.genres);
                    const candidateVector = genreList.map(genre => candidateGenres.has(genre) ? 1 : 0);
                    
                    let dotProduct = 0;
                    let magnitudeA = 0;
                    let magnitudeB = 0;

                    for (let i = 0; i < genreList.length; i++) {
                        dotProduct += likedVector[i] * candidateVector[i];
                        magnitudeA += likedVector[i] * likedVector[i];
                        magnitudeB += candidateVector[i] * candidateVector[i];
                    }
                    
                    // Calculate Cosine similarity
                    const score = (magnitudeA > 0 && magnitudeB > 0) 
                        ? dotProduct / (Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB)) 
                        : 0;
                    
                    return {
                        ...candidate,
                        score: score
                    };
                });
                
                // Step 5: Sort by score in descending order
                scoredMovies.sort((a, b) => b.score - a.score);
                
                // Step 6: Select top recommendations
                const topRecommendations = scoredMovies.slice(0, 2);
                
                // Step 7: Display results
                if (topRecommendations.length > 0) {
                    const recommendationTitles = topRecommendations.map(movie => movie.title);
                    resultElement.textContent = `Because you liked "${likedMovie.title}", we recommend: ${recommendationTitles.join(', ')}`;
                    resultElement.className = 'success';
                } else {
                    resultElement.textContent = `No recommendations found for "${likedMovie.title}".`;
                    resultElement.className = 'error';
                }
            } catch (error) {
                console.error('Error in recommendation calculation:', error);
                resultElement.textContent = "An error occurred while calculating recommendations.";
                resultElement.className = 'error';
            }
        }, 100);
    } catch (error) {
        console.error('Error in getRecommendations:', error);
        resultElement.textContent = "An unexpected error occurred.";
        resultElement.className = 'error';
    }
}
