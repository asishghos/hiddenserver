import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

export const extractClothingKeywords = async (req, res) => {
  const { paragraph } = req.body;

  if (!paragraph) {
    return res.status(400).json({ error: "Paragraph is required." });
  }

  try {
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
    //gemini-2.5-flash

    const prompt = `
You are an AI trained to extract **clothing-related keywords** from a paragraph.
Return only a clean list of **single or compound keywords** (max 3 words) related to clothing like 'blue denim jacket', 'cotton shirt', 'kurta', 'jeans', etc.

Example:
Input: "I wore a light cotton kurta with denim jeans and brown sandals during summer."
Output: ["cotton kurta", "denim jeans", "brown sandals"]

Now extract keywords from this:
"${paragraph}"
`;

    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();

    // Try to parse response if it's JSON-like
    let keywords = text;
    try {
      const parsed = JSON.parse(text);
      keywords = Array.isArray(parsed) ? parsed : [parsed];
    } catch (err) {
      // Fallback: clean basic Markdown list or comma-separated
      keywords = text
        .split(/[\n,-]/)
        .map((line) => line.replace(/[\*\[\]\"]+/g, "").trim())
        .filter((line) => line.length > 0);
    }

    res.json({ keywords });
  } catch (error) {
    console.error("Gemini error:", error.message);
    res.status(500).json({ error: "Failed to extract keywords from Gemini." });
  }
};
