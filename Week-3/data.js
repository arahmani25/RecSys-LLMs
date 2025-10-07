/* ===========================================================
   data.js
   - Loads and parses the MovieLens 100K dataset (u.item & u.data)
   - Provides movie list, user IDs, and rating data
   =========================================================== */

let movies = {}; // movieId -> movieTitle
let ratings = []; // {userId, itemId, rating}
let numUsers = 0;
let numMovies = 0;

const BASE_URL = "https://files.grouplens.org/datasets/movielens/ml-100k";

/**
 * Load the u.item and u.data files, parse them, and prepare
 * the global variables numUsers, numMovies, movies[], and ratings[].
 */
async function loadData() {
  const [itemText, dataText] = await Promise.all([
    fetch(`${BASE_URL}/u.item`).then(r => r.text()),
    fetch(`${BASE_URL}/u.data`).then(r => r.text())
  ]);

  parseItemData(itemText);
  parseRatingData(dataText);

  // Count unique users and movies
  const users = new Set(ratings.map(r => r.userId));
  const items = new Set(ratings.map(r => r.itemId));
  numUsers = users.size;
  numMovies = items.size;

  console.log(`Loaded ${numUsers} users, ${numMovies} movies`);
}

/**
 * Parse the u.item file: "movie id | title | ..."
 */
function parseItemData(text) {
  const lines = text.split("\n");
  lines.forEach(line => {
    if (!line.trim()) return;
    const parts = line.split("|");
    const id = parseInt(parts[0]);
    const title = parts[1];
    movies[id] = title;
  });
}

/**
 * Parse the u.data file: "user id \t item id \t rating \t timestamp"
 */
function parseRatingData(text) {
  const lines = text.split("\n");
  lines.forEach(line => {
    if (!line.trim()) return;
    const [userId, itemId, rating] = line.split("\t").map(Number);
    ratings.push({ userId, itemId, rating });
  });
}
