// data.js

// 1. GLOBAL STORAGE
let movies = [];

// 2. CONFIGURATION: The "Master Lists" (Stage 3 of your tutorial)
const MASTER_THEMES = [
    "Good vs Evil", "Coming of Age", "Redemption", "Sacrifice", 
    "Identity", "Survival", "Love", "Betrayal", "Underdog", 
    "Friendship", "Revenge", "Mystery"
];

const MASTER_SUBGENRES = [
    "Space Opera", "Cyberpunk", "Slasher", "High Fantasy", 
    "Police Procedural", "Romantic Comedy", "Dystopian", 
    "Zombie Apocalypse", "Adventure", "Drama"
];

// 3. SIMULATED LLM EXTRACTION (Stage 2)
// In a real app, this would call an API. Here, we check keywords.
function extractFeaturesFromText(text) {
    if (!text) return { themes: [], subgenres: [] };
    
    const lowerText = text.toLowerCase();
    const themes = [];
    const subgenres = [];

    // Check for Themes
    MASTER_THEMES.forEach(theme => {
        if (lowerText.includes(theme.toLowerCase())) {
            themes.push(theme);
        }
    });

    // Check for Sub-genres (and map some synonyms for "Consolidation")
    MASTER_SUBGENRES.forEach(genre => {
        if (lowerText.includes(genre.toLowerCase())) {
            subgenres.push(genre);
        }
    });

    // Fallback: If no themes found, tag as "General" so it's not empty
    if (themes.length === 0 && subgenres.length === 0) {
        themes.push("General");
    }

    return { themes, subgenres };
}

// 4. ENCODING (Stage 4: Create Vectors)
function oneHotEncode(items, masterList) {
    return masterList.map(masterItem => 
        items.includes(masterItem) ? 1 : 0
    );
}

// 5. CSV PARSER (Handles quoted strings in descriptions)
function parseCSV(text) {
    const rows = [];
    let currentRow = [];
    let currentField = '';
    let inQuotes = false;

    for (let i = 0; i < text.length; i++) {
        const char = text[i];
        
        if (char === '"') {
            if (inQuotes && text[i + 1] === '"') {
                currentField += '"'; // Handle escaped quotes
                i++;
            } else {
                inQuotes = !inQuotes; // Toggle quote state
            }
        } else if (char === ',' && !inQuotes) {
            currentRow.push(currentField);
            currentField = '';
        } else if ((char === '\n' || char === '\r') && !inQuotes) {
            if (currentField || currentRow.length > 0) {
                currentRow.push(currentField);
                rows.push(currentRow);
                currentRow = [];
                currentField = '';
            }
        } else {
            currentField += char;
        }
    }
    return rows;
}

// 6. MAIN LOAD FUNCTION
async function loadData() {
    try {
        // Fetch the raw CSV file
        const response = await fetch('movies_metadata.csv');
        if (!response.ok) throw new Error(`CSV not found: ${response.status}`);
        
        const text = await response.text();
        
        // Parse CSV
        const rawRows = parseCSV(text);
        
        // Find column indices (CSV headers are usually row 0)
        const headers = rawRows[0];
        const titleIdx = headers.indexOf('title');
        const overviewIdx = headers.indexOf('overview');
        const idIdx = headers.indexOf('id'); // TMDB ID

        // Process rows (Skip header, limit to top 500 for performance)
        // We start at index 1 to skip headers
        const processingLimit = Math.min(rawRows.length, 500); 
        
        for (let i = 1; i < processingLimit; i++) {
            const row = rawRows[i];
            
            // specific checks to ensure row has data
            if (!row[titleIdx] || !row[overviewIdx]) continue;

            const title = row[titleIdx];
            const overview = row[overviewIdx];
            const id = row[idIdx];

            // A. Extract
            const features = extractFeaturesFromText(overview);

            // B. Encode
            const themeVector = oneHotEncode(features.themes, MASTER_THEMES);
            const genreVector = oneHotEncode(features.subgenres, MASTER_SUBGENRES);
            const combinedVector = themeVector.concat(genreVector);

            movies.push({
                id: id,
                title: title,
                overview: overview,
                features: features,
                vector: combinedVector
            });
        }
        
        console.log(`Processed ${movies.length} movies.`);

    } catch (error) {
        console.error("Pipeline Error:", error);
        // Pass error to UI
        const resultElement = document.getElementById('result');
        if (resultElement) {
            resultElement.textContent = "Error loading data. Check console.";
            resultElement.className = 'error';
        }
    }
}
