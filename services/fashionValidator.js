import { ai } from "../index.js";
import { createUserContent } from "@google/genai";
import {
  uploadOnCloudinary,
  deleteFromCloudinary,
} from "../utils/cloudinary.js";

// Utility: convert an image URL to base64 inline data
async function fetchImageAsBase64(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(
        `Failed to fetch image: ${response.status} ${response.statusText}`
      );
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

// Utility: extract JSON from text response with proper bracket matching
function extractJsonFromText(text) {
  const startIndex = text.indexOf("{");
  if (startIndex === -1) return null;

  let braceCount = 0;
  let endIndex = startIndex;

  for (let i = startIndex; i < text.length; i++) {
    if (text[i] === "{") {
      braceCount++;
    } else if (text[i] === "}") {
      braceCount--;
      if (braceCount === 0) {
        endIndex = i;
        break;
      }
    }
  }

  if (braceCount !== 0) return null; // Unmatched braces

  const jsonString = text.substring(startIndex, endIndex + 1);
  return [jsonString]; // Return in array format to match the original regex match
}

export async function uploadAndValidateWithCritique(files, occasion, req) {
  const imageUrls = [];

  const userBodyShape = req.user.userBodyInfo.bodyShape || "";
  const userUnderTone = req.user.userBodyInfo.undertone || "";

  // Set default height for women if not provided (average women's height: 5'4")
  const userHeight =
    req.user.userBodyInfo.height &&
    (req.user.userBodyInfo.height.feet > 0 ||
      req.user.userBodyInfo.height.inches > 0)
      ? req.user.userBodyInfo.height
      : { feet: 5, inches: 4 };

  // Upload all images first
  for (const file of files) {
    const result = await uploadOnCloudinary(file.path);
    imageUrls.push(result);
  }

  // Single API call for both validation and critique using Gemini
  const validationResult = await validateAndCritiqueOutfitWithGemini(
    imageUrls,
    occasion,
    userBodyShape,
    userUnderTone,
    userHeight
  );

  // If any items are invalid fashion items, delete uploaded images and return error
  if (validationResult.hasInvalidFashionItems) {
    const deletionPromises = imageUrls.map((url) => deleteFromCloudinary(url));
    await Promise.all(deletionPromises);

    return {
      error: validationResult.invalidItemsMessage,
      imageUrls,
      deleted: true,
    };
  }

  // Calculate bad item images if there's a mismatch
  const badItemImages = validationResult.badItemIndices?.length
    ? validationResult.badItemIndices.map((idx) => imageUrls[idx])
    : [];

  return {
    imageUrls,
    critique: validationResult.critique,
    isPerfectMatch: validationResult.isPerfectMatch,
    badItemImages,
    suitabilityDetails: validationResult.suitabilityDetails,
  };
}

async function validateAndCritiqueOutfitWithGemini(
  imageUrls,
  occasion,
  bodyShape,
  undertone,
  height
) {
  const [top, bottom, accessory, footwear] = imageUrls;
  const labels = ["Topwear", "Bottomwear", "Accessory", "Footwear"];

  // Format height for better readability
  const heightString = `${height.feet}'${height.inches}"`;

  // Create user profile section for the prompt
  const userProfile = `
**USER PROFILE:**
- Body Shape: ${bodyShape || "Not specified"}
- Undertone: ${undertone || "Not specified"}  
- Height: ${heightString}`;

  try {
    const prompt = `You are a professional fashion stylist and image validation assistant. The user has uploaded 1 to 4 items labeled as: Topwear, Bottomwear, Accessory, and Footwear. The occasion is: ${occasion}.
${userProfile}
Your tasks:
**STEP 1: FASHION ITEM VALIDATION**
First, validate each uploaded image to ensure it contains a valid fashion item (clothing, footwear, or accessories). If ANY item is not a fashion item, respond with:
❌ INVALID FASHION ITEMS: [List the non-fashion items by their labels]
**STEP 2: OUTFIT CRITIQUE** (Only if all items are valid fashion items)
If all items are valid fashion items, provide:
1. A personalized fashion critique (within 60 words) considering the user's body shape, height, and how well the items work together for the given occasion. Focus on:
   - How the outfit flatters the occasion
   - How the outfit flatters the user's body shape
   - Proportions suitable for the user's height
   - Appropriateness for the occasion
2. Conclude with either:
   ✅ Perfect Match
   ❌ Not Suitable
3. **ALWAYS** provide the following JSON format for all items (regardless of Perfect Match or Not Suitable), including only the items provided:
{
  "Topwear": {
    "status": "suitable" | "not suitable",
    "reasoning": "Brief one-line explanation"
  },
  "Bottomwear": {
    "status": "suitable" | "not suitable",
    "reasoning": "Brief one-line explanation"
  },
  "Footwear": {
    "status": "suitable" | "not suitable",
    "reasoning": "Brief one-line explanation"
  },
  "Accessory": {
    "status": "suitable" | "not suitable",
    "reasoning": "Brief one-line explanation"
  }
}
⚠️ Rules:
- ALWAYS do fashion validation first according to the occasion
- Consider the user's body profile for personalized advice
- Only include keys in JSON for items that are provided
- The critique must come **before** the JSON
- **ALWAYS include the JSON structure for ALL items, whether Perfect Match or Not Suitable**
- Do not include any extra commentary outside the JSON
- The JSON must be valid and appear exactly as shown
- Each reasoning should be a concise one-liner explaining why the item is suitable/not suitable
Respond strictly in this structure.`;



    // Convert images to base64 format (filter out undefined/null URLs)
    const validImageUrls = imageUrls.filter((url) => url);
    const imageParts = await Promise.all(
      validImageUrls.map(fetchImageAsBase64)
    );

    // Create content parts - starting with text prompt
    const parts = [{ text: prompt }];

    // Add image parts
    for (const image of imageParts) {
      parts.push({
        inlineData: {
          mimeType: image.mimeType,
          data: image.data,
        },
      });
    }

    const contents = createUserContent(parts);

    const result = await ai.models.generateContent({
      model: "gemini-2.0-flash-thinking-exp", // Using Gemini's reasoning model
      contents: [contents],
      config: {
        temperature: 0.1, // Lower temperature for more consistent responses
        topP: 0.8,
        topK: 40,
      },
    });

    const responseText = result.candidates[0].content.parts[0].text;

    // Check if there are invalid fashion items
    if (responseText.includes("❌ INVALID FASHION ITEMS")) {
      const invalidItemsMatch = responseText.match(
        /❌ INVALID FASHION ITEMS:\s*\[([^\]]+)\]/i
      );
      const invalidItems = invalidItemsMatch
        ? invalidItemsMatch[1]
        : "Unknown items";

      return {
        hasInvalidFashionItems: true,
        invalidItemsMessage: `The following items are not valid fashion items: ${invalidItems}`,
      };
    }

    // Parse outfit critique
    const isPerfectMatch = responseText.includes("✅ Perfect Match");
    let badItemIndices = [];
    let suitabilityDetails = null;

    // Try to extract JSON for both suitable and unsuitable items
    const jsonMatch = extractJsonFromText(responseText);
    if (jsonMatch) {
      try {
        suitabilityDetails = JSON.parse(jsonMatch[0]);

        if (!isPerfectMatch) {
          // Get indices of unsuitable items - now checking the nested status
          badItemIndices = labels
            .map((label, index) => ({ label, index }))
            .filter(
              ({ label, index }) =>
                imageUrls[index] &&
                suitabilityDetails[label] &&
                suitabilityDetails[label].status === "not suitable"
            )
            .map(({ index }) => index);
        }
      } catch (error) {
        console.error("Failed to parse suitability JSON:", error);
        // Fallback: if JSON parsing fails and it's not a perfect match, assume all items are problematic
        if (!isPerfectMatch) {
          badItemIndices = imageUrls
            .map((url, index) => (url ? index : -1))
            .filter((index) => index !== -1);
        }
      }
    } else if (isPerfectMatch) {
      // If it's a perfect match but no JSON found, create default suitable entries
      suitabilityDetails = {};
      imageUrls.forEach((url, index) => {
        if (url && labels[index]) {
          suitabilityDetails[labels[index]] = {
            status: "suitable",
            reasoning: "Perfect match for the occasion and your profile",
          };
        }
      });
    }

    return {
      hasInvalidFashionItems: false,
      critique: responseText,
      isPerfectMatch,
      badItemIndices,
      suitabilityDetails,
    };
  } catch (error) {
    console.error("Error in Gemini fashion validation:", error);
    throw error;
  }
}
