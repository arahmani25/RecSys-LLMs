// data.js
// Responsible for loading and parsing MovieLens 100K u.item and u.data

// Public variables that script.js will use
let movies = {};        // movieId (number) -> title (string)
let ratings = [];       // array of {userId, itemId, rating}
let numUsers = 0;
let numMovies = 0;
let userIds = [];       // array of unique user IDs (1-indexed)
let movieIds = [];      // array of unique movie IDs (1-indexed)
let ratingValues = [];  // array of rating numbers (float)

const ML_BASE = 'https://files.grouplens.org/datasets/movielens/ml-100k';
const ITEM_URL = ML_BASE + '/u.item';
const DATA_URL = ML_BASE + '/u.data';

// loadData: fetches u.item and u.data, parses them, and fills global variables
async function loadData() {
  try {
    // Fetch both files in parallel
    const [itemResp, dataResp] = await Promise.all([
      fetch(ITEM_URL),
      fetch(DATA_URL)
    ]);

    if (!itemResp.ok || !dataResp.ok) {
      throw new Error('Failed to fetch MovieLens files. Check network or URL availability.');
    }

    const [itemText, dataText] = await Promise.all([
      itemResp.text(),
      dataResp.text()
    ]);

    parseItemData(itemText);
    parseRatingData(dataText);

    // compute unique counts
    const uniqueUsers = new Set(ratings.map(r => r.userId));
    const uniqueMovies = new Set(ratings.map(r => r.itemId));

    userIds = Array.from(uniqueUsers).sort((a,b) => a - b);
    movieIds = Array.from(uniqueMovies).sort((a,b) => a - b);

    numUsers = userIds.length;
    numMovies = movieIds.length;

    // ratingValues parallel to ratings array
    ratingValues = ratings.map(r => r.rating);

    console.log(`Loaded ${numUsers} users, ${numMovies} movies, ${ratings.length} ratings.`);
    return { numUsers, numMovies, ratings, movies };
  } catch (err) {
    console.error('loadData error:', err);
    throw err;
  }
}

// parseItemData(text)
// u.item lines are pipe-delimited. Format (first fields):
// movie id | movie title | release date | video release date | IMDb URL | ...genre flags...
function parseItemData(text) {
  movies = {};
  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    if (!line.trim()) continue;
    // some movie titles can contain pipes in older distributions, but in ML-100K it's safe to split
    const parts = line.split('|');
    const id = parseInt(parts[0], 10);
    const title = parts[1] ? parts[1].trim() : `Movie ${id}`;
    if (!Number.isNaN(id)) {
      movies[id] = title;
    }
  }
}

// parseRatingData(text)
// u.data format: user id \t item id \t rating \t timestamp
function parseRatingData(text) {
  ratings = [];
  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    if (!line.trim()) continue;
    const parts = line.split(/\t/);
    if (parts.length < 3) continue;
    const userId = parseInt(parts[0], 10);
    const itemId = parseInt(parts[1], 10);
    const rating = parseFloat(parts[2]);
    if ([userId, itemId, rating].some(v => Number.isNaN(v))) continue;
    ratings.push({ userId, itemId, rating });
  }
}
