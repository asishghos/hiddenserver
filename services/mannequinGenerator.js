import { ai } from "../index.js";
import { createUserContent, Modality } from "@google/genai";
import { getImageUrlFromBase64 } from "../services/base64ToImageUrl.js";

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

export async function generateModelImage(
  imageUrls,
  occasion,
  badItemImages = [],
  description = "",
  req
) {
  try {
    if (!Array.isArray(imageUrls) || imageUrls.length < 1) {
      throw new Error("At least one clothing item image URL is required.");
    }

    const [top, bottom, accessory, footwear] = imageUrls;
    const labels = ["Topwear", "Bottomwear", "Accessory", "Footwear"];

    const userBodyShape = req.user.userBodyInfo.bodyShape || "";
    const userUnderTone = req.user.userBodyInfo.undertone || "";

    // Set default height for women if not provided (average women's height: 5'4")
    const userHeight =
      req.user.userBodyInfo.height &&
      (req.user.userBodyInfo.height.feet > 0 ||
        req.user.userBodyInfo.height.inches > 0)
        ? req.user.userBodyInfo.height
        : { feet: 5, inches: 4 };

    // Format height for better readability
    const heightString = `${userHeight.feet}'${userHeight.inches}"`;

    // Build user profile section for the prompt
    const userProfile = `
👤 USER PROFILE FOR MODEL REPRESENTATION:
- Body Shape: ${userBodyShape || "Balanced proportions"}
- Height: ${heightString}
- Skin Undertone: ${userUnderTone || "Neutral"}

🎯 MODEL REQUIREMENTS:
- Generate a realistic female model that represents the user's body type and proportions
- Show how the outfit would look on someone with the user's specific body shape
- Use the undertone information for better color coordination in styling
- Ensure the model's height and body proportions match the user's profile for accurate visualization`;

    // Filter out bad items - only use good items for AI generation
    const goodItems = imageUrls
      .map((url, index) => ({
        url,
        label: labels[index],
        isGood: url && !badItemImages.includes(url),
      }))
      .filter((item) => item.url && item.isGood);

    // Identify what items we have vs what we need AI to generate
    const hasTop = goodItems.some((item) => item.label === "Topwear");
    const hasBottom = goodItems.some((item) => item.label === "Bottomwear");
    const hasAccessory = goodItems.some((item) => item.label === "Accessory");
    const hasFootwear = goodItems.some((item) => item.label === "Footwear");

    // Build dynamic prompt based on what user provided vs what AI needs to generate
    let itemInstructions = "OUTFIT COMPOSITION:\n";

    if (hasTop) {
      itemInstructions +=
        "• Topwear: USE the exact topwear shown in the reference image - match its style, color, and design precisely\n";
    } else {
      itemInstructions += `• Topwear: GENERATE appropriate topwear for ${occasion} that complements the provided items\n`;
    }

    if (hasBottom) {
      itemInstructions +=
        "• Bottomwear: USE the exact bottomwear shown in the reference image - match its style, color, and design precisely\n";
    } else {
      itemInstructions += `• Bottomwear: GENERATE appropriate bottomwear for ${occasion} that complements the provided items\n`;
    }

    if (hasFootwear) {
      itemInstructions +=
        "• Footwear: USE the exact footwear shown in the reference image - match its style, color, and design precisely\n";
    } else {
      itemInstructions += `• Footwear: GENERATE appropriate footwear for ${occasion} that complements the provided items\n`;
    }

    if (hasAccessory) {
      itemInstructions +=
        "• Accessories: USE the exact accessories shown in the reference image - match their style, color, and design precisely\n";
    } else {
      itemInstructions += `• Accessories: GENERATE tasteful accessories for ${occasion} that enhance the overall look\n`;
    }

    // Build description section for the prompt
    const descriptionSection =
      description && description.trim()
        ? `\nADDITIONAL STYLING REQUIREMENTS:\n${description.trim()}\n- Incorporate these specific preferences while maintaining the overall styling guidelines and occasion appropriateness\n- Balance user preferences with the provided reference items and occasion requirements\n`
        : "";

    const prompt = `You are an expert fashion stylist AI and professional image generator specializing in creating sophisticated, occasion-appropriate outfits with realistic model representation.

Generate a high-quality, full-body image of a female model styled for: ${occasion}.

${userProfile}

${itemInstructions}
${descriptionSection}
CRITICAL INSTRUCTIONS:
- For items with reference images: REPLICATE them exactly as shown (colors, patterns, style, fit)
- For missing items: CREATE complementary pieces that work harmoniously with the provided items
- Ensure all generated items are appropriate for the specified occasion: ${occasion}
- Maintain color coordination and style consistency across all items
- Consider the user's undertone (${
      userUnderTone || "neutral"
    }) when selecting colors for generated items
- If only some items are provided, make sure the AI-generated items complement and enhance the user's choices
${
  description && description.trim()
    ? `- Incorporate the user's styling preferences: "${description.trim()}" while maintaining occasion appropriateness`
    : ""
}

VISUAL SPECIFICATIONS:
- Generate a realistic **Indian** female model with ${
      userBodyShape || "balanced"
    } body shape and ${heightString} height proportions
- The model should represent how the outfit would realistically look on someone with the user's body type
- Model pose: Confident, natural stance with good posture that showcases the outfit on the specific body shape
- Styling: Ensure proper fit that flatters the user's body shape, with attention to proportions
- Aesthetic: Modern, polished, and fashion-forward appearance suitable for the user's body type
- Background: Clean, minimalist backdrop in neutral tones (white, light gray, or soft beige)
- Lighting: Professional studio lighting with soft shadows to highlight outfit details and body silhouette
- Image quality: High-resolution, sharp focus on clothing details and overall composition

BODY-SPECIFIC STYLING FOCUS:
- Demonstrate how the outfit flatters the ${
      userBodyShape || "balanced"
    } body shape
- Show proper proportions and fit for someone of ${heightString} height
- Ensure the styling choices work well with the user's body type for a realistic preview
- Focus on creating a look that the user would actually achieve with their body characteristics

DESIGN PRINCIPLES:
- Maintain elegance and sophistication appropriate for ${occasion}
- Ensure color harmony that works with ${userUnderTone || "neutral"} undertones
- Consider balanced proportions between provided and generated items for the user's body type
- Consider current fashion trends while maintaining timeless appeal
- Focus on creating a cohesive, well-curated look that works specifically for the user's body shape
- The final outfit should look intentionally styled for the user's body type, not generic styling
${
  description && description.trim()
    ? `- Reflect the user's personal style preferences: "${description.trim()}" in the overall aesthetic`
    : ""
}

**For Indian Festive/Traditional Occasions:**
- Include EXACTLY ONE complete ethnic outfit with:
  - One main garment: Saree with blouse, Lehenga choli, Anarkali suit, Sharara set, Palazzo suit, Indo-western fusion wear, or Churidar kurta set
  - Traditional footwear: Juttis, kolhapuris, wedges, or embellished heels
  - 2–4 traditional accessories: Statement jewelry (necklace, earrings, bangles), potli bag/clutch, dupatta (if applicable), maang tikka, nose ring, or traditional hair accessories
  - Optional: Traditional makeup elements like bindi, kajal, or mehendi patterns

**Ethnic Wear Styling Guidelines:**
- Focus on rich fabrics: silk, brocade, chiffon, georgette, velvet, or cotton with traditional prints
- Incorporate traditional Indian colors: deep jewel tones, metallics, vibrant festival colors, or elegant pastels
- Include authentic Indian embellishments: zardozi, mirror work, thread embroidery, sequins, or block prints
- Ensure proper draping and fit for traditional garments
- Balance traditional elements with contemporary styling where appropriate

**For Contemporary/Western Occasions:**
- Include EXACTLY ONE complete outfit with:
  - One topwear (shirt, blouse, t-shirt, sweater, etc.)
  - One bottomwear (pants, skirt, shorts, etc.) OR a dress (if dress, no separate top/bottom needed)
  - One pair of shoes
  - 2–3 accessories (bag, hat, jewelry, belt, scarf – choose what fits the occasion)

PERSONALIZATION GOAL:
Create a realistic visualization showing how the complete outfit would look on someone with the user's exact body specifications (${
      userBodyShape || "balanced"
    } shape, ${heightString} height, ${
      userUnderTone || "neutral"
    } undertone), providing an accurate and personalized preview of the styled look.

Style the model to embody confidence and grace while accurately representing the user's body type, showcasing how this complete outfit would realistically look for ${occasion}${
      description && description.trim()
        ? ` with the requested styling approach`
        : ""
    }.`;

    // Convert only the good items to base64 format
    const imageParts = await Promise.all(
      goodItems.map((item) => fetchImageAsBase64(item.url))
    );

    // Modify the prompt to include description request
    const parts = [{ text: prompt }];

    // Add image parts for the good items only
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
      model: "gemini-2.0-flash-exp",
      contents: [contents],
      config: {
        responseModalities: ["TEXT", "IMAGE"],
        temperature: 0.7, // Slightly higher for more creative styling
        topP: 0.9,
      },
    });

    // After generating, extract both image and description
    const resultPart = result.candidates[0].content.parts;

    const imagePart = resultPart.find((p) => p.inlineData);

    if (imagePart && imagePart.inlineData?.data) {
      console.log("image create successfully");
      return {
        type: "uploaded_image",
        imageUrl: await getImageUrlFromBase64(
          imagePart.inlineData.data,
          req.user._id
        ),
      };
    } else {
      console.error("No image data returned from AI model");
      return {
        type: "uploaded_image",
        imageUrl: null,
      };
    }
  } catch (error) {
    console.error("Error generating model image:", error);
    return {
      type: "uploaded_image",
      imageUrl: null,
    };
  }
}
