import axios from "axios";

// Configuration
const SCRAPINGDOG_API_KEY = "68419872721898288f2db21d";
const DELAY_BETWEEN_REQUESTS = 5000; // 5 seconds between requests
const MAX_RETRIES = 3;
const RETRY_DELAY = 10000; // 10 seconds base retry delay

// Main controller function
export const getProducts = async (req, res) => {
  try {
    const { keywords } = req.body;
    
    // Validate input
    if (!keywords || !keywords.length || !Array.isArray(keywords[0])) {
      return res.status(400).json({ error: "Invalid or missing keywords" });
    }

    // Search products using ScrapingDog API
    const results = await searchWithScrapingDog(keywords);
    
    res.json(results);
  } catch (error) {
    console.error("Error in /products:", error);
    res.status(500).json({ 
      error: error.message || "Failed to fetch product results" 
    });
  }
};

// ScrapingDog API search function
export const searchWithScrapingDog = async (keywords) => {
  const results = [];

  for (let i = 0; i < keywords.length; i++) {
    const keyword = keywords[i][0];
    
    try {
      const query = `${keyword} for women`;
      // console.log(`Searching for: "${query}"`);

      const searchResults = await performScrapingDogSearch(query);
      const convertedResults = convertScrapingDogResults(searchResults, keyword);

      // console.log(`Found ${searchResults.length} results for "${keyword}"`);
      
      // Add top 3 results
      results.push(...convertedResults.slice(0, 3));

      // Add delay between requests (except for the last one)
      if (i < keywords.length - 1) {
        // console.log(`Waiting ${DELAY_BETWEEN_REQUESTS}ms before next request...`);
        await delay(DELAY_BETWEEN_REQUESTS);
      }
    } catch (error) {
      console.error(`Error searching for ${keyword}:`, error);
    }
  }

  return results;
};

// ScrapingDog API call with retry logic
const performScrapingDogSearch = async (query, retryCount = 0) => {
  try {
    const response = await axios.get(
      "https://api.scrapingdog.com/google_shopping/",
      {
        params: {
          api_key: SCRAPINGDOG_API_KEY,
          query: query,
          results: 10,
          country: "in",
        },
        timeout: 15000, // 15 second timeout
      }
    );

    return response.data.shopping_results || [];
  } catch (error) {
    const status = error.response?.status;
    const message = error.response?.data?.error || error.message;

    console.error(`ScrapingDog API error (attempt ${retryCount + 1}):`, {
      status,
      message,
      query,
    });

    // Handle rate limiting and server errors with retry
    if (shouldRetry(status) && retryCount < MAX_RETRIES) {
      const delayTime = RETRY_DELAY * (retryCount + 1); // Exponential backoff
      console.log(`Retrying in ${delayTime}ms...`);
      await delay(delayTime);
      return performScrapingDogSearch(query, retryCount + 1);
    }

    // If max retries exceeded
    if (retryCount >= MAX_RETRIES) {
      throw new Error(
        `ScrapingDog API failed after ${MAX_RETRIES} retries for query: ${query}`
      );
    }

    throw error;
  }
};

// Convert ScrapingDog results to standardized format
const convertScrapingDogResults = (searchResults, keyword) => {
  return searchResults.map((item) => ({
    keyword,
    source: item.source || "scrapingdog",
    title: cleanTitle(item.title || ""),
    price: item.price || "",
    rating: item.rating || "",
    product_id: item.product_id || "",
  }));
};

// Clean up product titles
const cleanTitle = (title) => {
  return title
    .replace(
      /\s*-\s*(Buy|Shop|Online|Price|India|Amazon|Flipkart|Myntra|Ajio|Nykaa).*$/gi,
      ""
    )
    .replace(/\s*\|\s*.*$/, "")
    .replace(/\s*–\s*.*$/, "")
    .replace(/\s*\.\.\.$/, "")
    .trim();
};

// Utility function to check if error should be retried
const shouldRetry = (status) => {
  return status === 429 || status >= 500;
};

// Utility function for delays
const delay = (ms) => {
  return new Promise(resolve => setTimeout(resolve, ms));
};

// Utility function to check ScrapingDog API status
export const checkScrapingDogQuota = async () => {
  try {
    const response = await axios.get(
      "https://api.scrapingdog.com/google_shopping/",
      {
        params: {
          api_key: SCRAPINGDOG_API_KEY,
          query: "test",
          results: 1,
          country: "in",
        },
        timeout: 10000,
      }
    );
    
    console.log("ScrapingDog API is working. Quota check successful.");
    return true;
  } catch (error) {
    console.error(
      "ScrapingDog API quota check failed:",
      error.response?.data || error.message
    );
    return false;
  }
};