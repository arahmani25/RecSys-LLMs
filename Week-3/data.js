// ------------------------------
// DATA LOADING AND PARSING LOGIC
// ------------------------------

// URLs to your locally downloaded MovieLens dataset files
const itemURL = './u.item';
const ratingURL = './u.data';

// Arrays to hold parsed data
let movies = [];
let ratings = [];
let numUsers = 0;
let numMovies = 0;

// ------------------------------
// Load both movie and rating data
// ------------------------------
async function loadData() {
  try {
    const [itemResponse, ratingResponse] = await Promise.all([
      fetch(itemURL),
      fetch(ratingURL)
    ]);

    if (!itemResponse.ok || !ratingResponse.ok) throw new Error("Failed to fetch");

    const itemText = await itemResponse.text();
    const ratingText = await ratingResponse.text();

    parseItemData(itemText);
    parseRatingData(ratingText);

    numUsers = Math.max(...ratings.map(r => r.userId)) + 1;
    numMovies = Math.max(...ratings.map(r => r.movieId)) + 1;

    return true;
  } catch (err) {
    document.getElementById("result").textContent =
      "Error loading data or training model: " + err.message;
    throw err;
  }
}

// ------------------------------
// Parse u.item file (Movie info)
// ------------------------------
function parseItemData(text) {
  const lines = text.trim().split('\n');
  movies = lines.map(line => {
    const [id, title] = line.split('|');
    return { id: parseInt(id) - 1, title };
  });
}

// ------------------------------
// Parse u.data file (User ratings)
// ------------------------------
function parseRatingData(text) {
  const lines = text.trim().split('\n');
  ratings = lines.map(line => {
    const [userId, movieId, rating] = line.split('\t');
    return {
      userId: parseInt(userId) - 1,
      movieId: parseInt(movieId) - 1,
      rating: parseFloat(rating)
    };
  });
}
