import { ai } from "../index.js";
import { createUserContent } from "@google/genai";

// convert an image URL to base64 inline data
async function fetchImageAsBase64(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch image: ${response.status} ${response.statusText}`);
        }
        const buffer = await response.arrayBuffer();
        const contentType = response.headers.get("content-type") || "image/png";
        return {
            mimeType: contentType,
            data: Buffer.from(buffer).toString("base64"),
        };
    } catch (error) {
        console.error(`Error fetching image from ${url}:`, error);
        throw error;
    }
}

// Helper function to safely parse JSON from AI response
function parseJSONFromResponse(responseText) {
    try {
        // First try to parse as-is
        return JSON.parse(responseText);
    } catch (error) {
        // console.log("Direct JSON parse failed, trying to extract from markdown...");
        
        // Try to extract JSON from markdown code blocks
        const jsonMatch = responseText.match(/```json\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
            try {
                return JSON.parse(jsonMatch[1]);
            } catch (e) {
                console.log("Markdown JSON parse failed, trying to find JSON object...");
            }
        }
        
        // If no markdown blocks, try to find JSON-like content
        const jsonStart = responseText.indexOf('{');
        const jsonEnd = responseText.lastIndexOf('}');
        
        if (jsonStart !== -1 && jsonEnd !== -1 && jsonEnd > jsonStart) {
            const jsonStr = responseText.substring(jsonStart, jsonEnd + 1);
            try {
                return JSON.parse(jsonStr);
            } catch (e) {
                console.log("Extracted JSON parse failed");
            }
        }
        
        // If all else fails, log the response and throw error
        console.error("Failed to parse JSON from response:", responseText);
        throw new Error('No valid JSON found in response: ' + responseText.substring(0, 200));
    }
}

export async function validateChatbotImage(imageUrl, userQuery = '') {
    const titleGuidance = userQuery
        ? `\n\nIMPORTANT: Use the user query "${userQuery}" as the primary basis for the title. The title should be exactly 3-4 words long and directly reflect what the user is asking about or searching for. Expand or refine the user query if needed to make it 3-4 words.`
        : `\n\nSince no user query is provided, create a title that is exactly 3-4 words long based on the main clothing items visible.`;

    const prompt = `You are a professional fashion AI analyst. Analyze this image carefully and provide the following information:

ANALYSIS CRITERIA:
1. Fashion Items: Look for any clothing, footwear, accessories, or fashion-related items (including bags, jewelry, hats, etc.)
2. Full-Body Human: Determine if there's a complete human figure visible from head to toe (or head to feet if wearing shoes)
3. Fashion Title: If a full-body human is present, generate a title based on the user's query

RESPONSE FORMAT:
Respond ONLY with valid JSON. Do not include any markdown formatting, code blocks, or additional text. Return exactly this structure:
{
    "containsFashionItem": boolean,
    "containsFullBodyHuman": boolean,
    "generatedTitle": string or null
}

GUIDELINES:
- Set "containsFashionItem" to true if ANY fashion-related item is visible
- Set "containsFullBodyHuman" to true only if you can see a person's complete silhouette from head to feet
- Only provide "generatedTitle" if "containsFullBodyHuman" is true
- If no full-body human, set "generatedTitle" to null
- The title should be exactly 3-4 words long and derived from the user query when provided
- If the user query is shorter than 3 words, expand it contextually based on the image
- If the user query is longer than 4 words, condense it to the most essential 3-4 words${titleGuidance}

IMPORTANT: Your response must be ONLY valid JSON with no additional formatting or text.`;

    try {
        // Get the image data
        const imageData = await fetchImageAsBase64(imageUrl);

        const parts = [
            { text: prompt },
            {
                inlineData: imageData  // Use the imageData object directly
            }
        ];

        const contents = createUserContent(parts);
        const result = await ai.models.generateContent({
            model: 'gemini-2.0-flash-thinking-exp',
            contents: [contents],
            config: {
                temperature: 0.1,
                topP: 0.8,
                topK: 40,
            },
        });

        const responseText = result.candidates[0].content.parts[0].text;
        // console.log("Raw AI response:", responseText); // Debug log
        
        const response = parseJSONFromResponse(responseText);
        return response;
        
    } catch (error) {
        console.error("Error in validateChatbotImage:", error);
        
        // Return a default response in case of error
        return {
            containsFashionItem: false,
            containsFullBodyHuman: false,
            generatedTitle: null
        };
    }
}