import OpenAI from "openai";
import fetch from "node-fetch"; // if not using native fetch
import path from "path";
// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY, // Make sure to set your API key
});

// Converts a public image URL to a base64 data URL
async function convertImageUrlToBase64(imageUrl) {
  try {
    const response = await fetch(imageUrl);

    if (!response.ok) {
      throw new Error(
        `Failed to fetch image: ${response.status} ${response.statusText}`
      );
    }

    const contentType = response.headers.get("content-type") || "image/png";

    // Validate content type is image/*
    if (!contentType.startsWith("image/")) {
      throw new Error(`Unsupported content type: ${contentType}`);
    }

    const buffer = await response.arrayBuffer();
    const base64 = Buffer.from(buffer).toString("base64");

    return `data:${contentType};base64,${base64}`;
  } catch (error) {
    console.error("Error converting image URL to base64:", error.message);
    throw error;
  }
}

export async function analyzeGeneratedAndBadImages(
  results,
  badImages,
  occasion
) {
  try {
    const prompt = `
You are a professional fashion AI analyst. Analyze the provided images and provide detailed insights.

CONTEXT:
- Occasion: "${occasion}"
- You will receive TWO types of images in sequence:
  1. GENERATED IMAGES (AI-created fashion outfits) - These come first
  2. BAD/UNSUITABLE IMAGES (inappropriate items for the occasion) - These come after

IMAGE ORDER EXPLANATION:
- The first ${
      results?.length || 0
    } images are AI-GENERATED fashion outfits suitable for "${occasion}"
- The remaining ${
      badImages?.length || 0
    } images are BAD/UNSUITABLE items for "${occasion}"

ANALYSIS REQUIREMENTS:

FOR GENERATED IMAGES (First ${results?.length || 0} images):
- Provide a brief, engaging description (1-3 lines maximum)
- Focus on style, fit, and occasion-appropriateness
- Describe the overall look and key fashion elements

FOR BAD IMAGES (Remaining ${badImages?.length || 0} images)

KEYWORDS GENERATION:
- goodImagesKeywords: Combine all relevant fashion keywords from the generated images (garments, footwear, accessories, bags, jewelry)
- badImagesKeywords: Suggest alternative fashion items that WOULD be suitable for "${occasion}" instead of the bad items shown

RESPONSE FORMAT:
Respond only in valid JSON format:
{
  "goodImageWithDescription": [
    {
      "image": "url of the image",
      "description": "Brief description (1-3 lines) of the image"
    },
    ...
  ],
  "badImageWithDescription": [
    {
      "image": "url of the image",
      "description": "Brief description (1-3 lines) of the image"
    },
    ...
  ],
  "goodImagesKeywords": ["keyword1", "keyword2", "..."],
  "badImagesKeywords": ["suitable_alternative1", "suitable_alternative2", "..."]
}

GUIDELINES:
- Keywords must be fashion items only (clothing, shoes, accessories, bags, jewelry)
- Keep descriptions concise (1-3 lines maximum)
- Use specific fashion terminology
- goodImagesKeywords should reflect items seen in generated images
- badImagesKeywords should suggest better alternatives for the occasion

Analyze the images now:
`;

    const messages = [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: prompt,
          },
        ],
      },
    ];

    // Add generated images (results array)
    if (results && results.length > 0) {
      for (let i = 0; i < results.length; i++) {
        const result = results[i];
        if (result && result.imageUrl != null) {
          // Handle different possible image formats
          let imageUrl = null;

          if (typeof result === "string") {
            // If result is directly a base64 string or URL
            imageUrl = result;
          } else if (result.imageUrl) {
            imageUrl = result.imageUrl;
          } else if (result.imageB64) {
            imageUrl = result.imageB64;
          } else if (result.data && result.data.imageUrl) {
            imageUrl = result.data.imageUrl;
          } else if (result.data && result.data.imageB64) {
            imageUrl = result.data.imageB64;
          }

          if (imageUrl) {
            try {
              const processedImage = await convertImageUrlToBase64(imageUrl);
              messages[0].content.push({
                type: "image_url",
                image_url: {
                  url: processedImage,
                  detail: "high",
                },
              });
            } catch (error) {
              console.error(`Error processing generated image ${i}:`, error);
            }
          }
        }
      }
    }

    // Add bad images
    if (Array.isArray(badImages) && badImages.length > 0) {
      for (let i = 0; i < badImages.length; i++) {
        const badImage = badImages[i];
        if (badImage && badImage.imageUrl) {
          try {
            const processedImage = await convertImageUrlToBase64(
              badImage.imageUrl
            );
            messages[0].content.push({
              type: "image_url",
              image_url: {
                url: processedImage,
                detail: "high",
              },
            });
          } catch (error) {
            console.error(`Error processing bad image ${i}:`, error);
          }
        }
      }
    }
    console.log("Results:", results);
    console.log("Bad Images:", badImages);

    if (messages[0].content.length === 1) {
      console.warn(
        "No valid images were added to the prompt. Returning empty result."
      );
      return {
        goodImageWithDescription: [],
        badImageWithDescription: [],
        goodImagesKeywords: [],
        badImagesKeywords: [],
      };
    }

    const completion = await openai.chat.completions.create({
      model: "gpt-4-turbo",
      messages: messages,
      temperature: 0.3,
      response_format: { type: "json_object" },
    });
    if (!completion.choices?.[0]?.message?.content) {
      throw new Error("OpenAI did not return content");
    }

    console.log(JSON.stringify(completion, null, 2));

    // After parsing the response from OpenAI
    const response = JSON.parse(completion.choices[0].message.content);

    // Merge back original URLs
    const allOriginalImages = [...results, ...badImages].map((img) => {
      if (typeof img === "string") return img;
      return (
        img.imageUrl || img.imageB64 || img.data?.imageUrl || img.data?.imageB64
      );
    });

    // Split original images based on counts
    const goodImageCount = results.length;
    const goodOriginalImages = allOriginalImages.slice(0, goodImageCount);
    const badOriginalImages = allOriginalImages.slice(goodImageCount);

    // Remap good images
    if (response.goodImageWithDescription) {
      response.goodImageWithDescription = response.goodImageWithDescription.map(
        (desc, index) => ({
          ...desc,
          image: goodOriginalImages[index] || desc.image,
        })
      );
    }

    // Remap bad images
    if (response.badImageWithDescription) {
      response.badImageWithDescription = response.badImageWithDescription.map(
        (desc, index) => ({
          ...desc,
          image: badOriginalImages[index] || desc.image,
        })
      );
    }

    return {
      goodImageWithDescription: response.goodImageWithDescription || [],
      badImageWithDescription: response.badImageWithDescription || [],
      goodImagesKeywords: response.goodImagesKeywords || [],
      badImagesKeywords: response.badImagesKeywords || [],
    };
  } catch (error) {
    console.error("Error in analyzeGeneratedAndBadImages:", error);
    throw error;
  }
}
