import { ai } from "../index.js";
import { DigitalWardrobe } from "../models/digitalWardrobe.models.js";
import { getImageUrlFromBase64 } from "../services/base64ToImageUrl.js";

export const getGarmentsFromWardrobe = async (req, occasion) => {
  try {
    const userId = req.user._id;

    if (!occasion) return [];

    const wardrobe = await DigitalWardrobe.findOne({ userId });

    if (!wardrobe || wardrobe.uploadedImages.length === 0) {
      return [];
    }

    const result = wardrobe.uploadedImages
      .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
      .flatMap((image) =>
        image.garments
          .filter((g) => g.occasion.includes(occasion))
          .map((g) => ({
            imageId: image._id,
            itemName: g.itemName,
            imageUrl: image.imageUrl,
            createdAt: image.createdAt,
          }))
      );

    return result;
  } catch (err) {
    console.error("Error fetching by occasion:", err);
    return [];
  }
};

export const convertImageToBase64 = async (imageUrl) => {
  try {
    // If imageUrl is already a base64 string or buffer
    if (typeof imageUrl === "string" && imageUrl.startsWith("data:image")) {
      return imageUrl.split(",")[1]; // Remove data:image/jpeg;base64, prefix
    }

    // If it's a buffer
    if (Buffer.isBuffer(imageUrl)) {
      return imageUrl.toString("base64");
    }

    // If it's a URL, fetch and convert
    if (
      typeof imageUrl === "string" &&
      (imageUrl.startsWith("http") || imageUrl.startsWith("https"))
    ) {
      const response = await fetch(imageUrl);
      const buffer = await response.arrayBuffer();
      return Buffer.from(buffer).toString("base64");
    }

    // If it's already base64 string
    if (typeof imageUrl === "string") {
      return imageUrl;
    }

    throw new Error("Unsupported image format");
  } catch (error) {
    console.error("Error converting image to base64:", error);
    throw error;
  }
};

export const selectWardrobeItem = (wardrobeItems, userId, occasion) => {
  if (wardrobeItems.length === 1) {
    return wardrobeItems[0];
  }

  // Simple rotation logic using timestamp-based selection
  const sessionKey = `${userId}_${occasion}`;
  const currentTime = Date.now();
  const index = Math.floor((currentTime / 1000) % wardrobeItems.length);

  return wardrobeItems[index];
};

export const generateWardrobeStyledImage = async (
  selectedGarment,
  occasion,
  description = "",
  req
) => {
  try {
    // Convert image to base64
    const base64Image = await convertImageToBase64(selectedGarment.imageUrl);

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
- Generate a female model that represents the user's body type and proportions
- Show how the outfit would look on someone with the user's specific body shape
- Use the undertone information for better color coordination in styling
- Ensure the model's height and body proportions match the user's profile`;

    // Build description section for the prompt
    const descriptionSection =
      description && description.trim()
        ? `\n📝 ADDITIONAL STYLING REQUIREMENTS:\n- ${description.trim()}\n- Incorporate these specific preferences while maintaining the overall styling guidelines\n`
        : "";

    // Enhanced occasion-specific styling guidelines
    const getOccasionGuidelines = (occasion) => {
      const guidelines = {
        casual: "Relaxed, comfortable styling with trendy accessories",
        business: "Professional, polished look with sophisticated accessories",
        formal: "Elegant, refined styling with luxury accessories",
        party: "Glamorous, eye-catching styling with statement accessories",
        date: "Chic, attractive styling with romantic touches",
        vacation: "Comfortable, stylish travel-appropriate styling",
        workout: "Athletic, functional styling with sporty accessories",
        diwali:
          "Focus on rich, festive colors like deep reds, golds, royal blues, or jewel tones. Include traditional jewelry, ethnic footwear, and celebratory elements. Ensure the outfit is formal enough for temple visits and family gatherings.",
        wedding:
          "Create an elegant, formal look with rich fabrics and traditional jewelry. Avoid white unless specifically requested. Include appropriate accessories for the level of formality.",
        "karva chauth":
          "Style with traditional elements, rich colors (especially red, maroon, or pink), and appropriate jewelry. Include sindoor, mehendi consideration, and festive accessories.",
        navratri:
          "Incorporate vibrant colors specific to the day if known, traditional jewelry, and comfortable styling for dancing. Focus on celebratory, colorful elements.",
        holi: "Light, comfortable fabrics in white or light colors that can handle color play. Minimal jewelry, practical footwear, and easy-to-wash elements.",
        "durga puja":
          "Traditional Bengali influences if appropriate, elegant styling with cultural elements, appropriate jewelry, and festive colors.",
        "ganesh chaturthi":
          "Bright, festive colors, traditional styling with cultural appropriateness, and celebratory elements.",
        casual:
          "Comfortable, relaxed styling with minimal accessories. Focus on comfort and ease of movement.",
        travel:
          "Comfortable, practical styling with minimal accessories. Focus on comfort and versatility.",
        religious:
          "Modest, culturally appropriate styling with traditional elements. Focus on respectful, conservative approach.",
      };
      return (
        guidelines[occasion?.toLowerCase()] ||
        "Versatile, well-coordinated styling"
      );
    };

    const wardrobePrompt = `
ROLE: You are a professional fashion stylist and AI image generator specializing in both contemporary and traditional Indian fashion, creating personalized styled outfit visualizations.

OBJECTIVE: 
Create a complete, professionally styled outfit for a **${occasion}** occasion, featuring the clothing item shown in the provided image as the main piece.
${userProfile}
${descriptionSection}

✅ STYLING REQUIREMENTS:
- Use the provided clothing item as the CENTRAL piece of the outfit
- Complete the look with complementary pieces based on garment type:
  
  **For Western/Contemporary Wear:**
  - Add appropriate additional garments (if it's a top, add bottom; if it's a bottom, add top; if it's a dress, minimal layering only)
  - Include suitable footwear for ${occasion}
  - Add 2-3 accessories that enhance the overall look (bag, jewelry, belt, scarf, etc.)
  
  **For Indian Ethnic/Traditional Wear:**
  - Saree: Complete with matching or contrasting blouse, petticoat (if visible), appropriate jewelry (necklace, earrings, bangles, maang tikka for festive occasions)
  - Lehenga: Style with matching or complementary choli/blouse, dupatta, traditional jewelry set
  - Salwar Kameez/Churidar: Complete with dupatta, matching bottoms, ethnic accessories
  - Kurti/Kurta: Pair with appropriate bottoms (palazzo, churidar, jeans for fusion), dupatta if needed
  - Ethnic tops: Style with traditional or fusion bottoms, appropriate jewelry
  - Include traditional footwear (juttis, mojaris, sandals) or fusion footwear as appropriate
  - Add ethnic accessories: jewelry (kundan, oxidized, gold-toned), potli bags, ethnic belts, hair accessories

- Ensure the entire outfit is perfect and appropriate for **${occasion}**
- ${getOccasionGuidelines(occasion)}

🎨 STYLING FOCUS:
- Make the provided garment look its absolute best on the user's body type
- Create a cohesive, well-coordinated look that flatters the user's body shape
- For ethnic wear, ensure cultural authenticity while maintaining modern styling sensibilities
- Use colors that complement both the garment and work well with the user's undertone
- Consider proportions that work well for the user's height (${heightString})
- Show how this personal wardrobe item can be styled beautifully for the user's specific body type
- For Indian festive occasions, incorporate appropriate traditional elements while keeping the look elegant and wearable

🖼️ VISUAL OUTPUT REQUIREMENTS:
- Generate a realistic female model with ${
      userBodyShape || "balanced"
    } body shape and ${heightString} height proportions
- The model should represent how the outfit would realistically look on someone with the user's body type
- **FULL-SIZE, COMPLETE OUTFIT VISUALIZATION**: Show the entire styled look from head to toe
- Include all styling elements: complete garment, accessories, footwear, jewelry, and any additional pieces
- Full-body shot with clean, neutral background (white, soft gray, or subtle ethnic-inspired backdrop for traditional wear)
- High-quality fashion photography style with professional lighting
- Confident, natural pose that showcases the complete outfit clearly
- Focus on demonstrating how the outfit flatters the specific body shape
- Ensure the model's proportions and body type match the user's profile for realistic visualization
- For ethnic wear, include appropriate hair styling and makeup that complements the traditional look

✨ PERSONALIZATION GOAL:
Create a realistic, complete visualization showing how the user's wardrobe item would look when styled into a full outfit on someone with their exact body specifications, providing an accurate preview of the complete styled look including all necessary components and accessories.

🇮🇳 INDIAN ETHNIC WEAR SPECIFICATIONS:
- Maintain cultural authenticity while ensuring modern styling appeal
- Consider regional variations in traditional wear when appropriate
- Include appropriate traditional makeup and hair styling suggestions in the visualization
- Ensure jewelry and accessories are proportionate to the occasion's formality
- For festive occasions, incorporate rich colors, traditional patterns, and celebratory elements
- Balance traditional elements with contemporary styling for a modern, elegant look
- Consider seasonal appropriateness for Indian climate and festivals

🔧 TECHNICAL SPECIFICATIONS:
- Photorealistic, high-resolution image generation
- Professional fashion photography lighting and composition
- Model body type and proportions that accurately represent the user's specifications
- Clear focus on the provided garment while showing the complete styled outfit with ALL components
- Background that enhances but doesn't distract from the styling
- Generate consistent, realistic human model based on the provided body specifications
- Ensure all outfit elements are clearly visible and properly proportioned
- **MANDATORY**: Show complete outfit from head to toe including all accessories, footwear, and styling elements
`;
    const wardrobeContents = [
      { text: wardrobePrompt },
      {
        inlineData: {
          mimeType: "image/jpeg",
          data: base64Image,
        },
      },
    ];

    const wardrobeResponse = await ai.models.generateContent({
      model: "gemini-2.0-flash-preview-image-generation",
      contents: wardrobeContents,
      config: {
        responseModalities: ["TEXT", "IMAGE"],
        temperature: 0.7,
        topP: 0.9,
      },
    });

    for (const part of wardrobeResponse.candidates[0].content.parts) {
      if (part.inlineData) {
        return {
          success: true,
          data: {
            imageB64: part.inlineData.data,
          },
        };
      }
    }

    throw new Error("No image generated in AI response");
  } catch (error) {
    console.error("Error generating wardrobe styled image:", error);
    return {
      success: false,
      error: error.message,
      userMessage:
        "Unable to generate styled image. Please try again or contact support if the issue persists.",
    };
  }
};

export const processWardrobeItemForOccasion = async (
  req,
  occasion,
  description = ""
) => {
  try {
    // Get wardrobe items for the occasion
    const wardrobeItems = await getGarmentsFromWardrobe(req, occasion);
    if (wardrobeItems.length === 0) {
      return {
        type: "wardrobe",
        imageB64: null,
      };
    }

    // Select a wardrobe item using rotation logic
    const selectedGarment = selectWardrobeItem(
      wardrobeItems,
      req.user._id,
      occasion
    );

    // Generate styled image with optional description
    const result = await generateWardrobeStyledImage(
      selectedGarment,
      occasion,
      description,
      req
    );

    if (result.success) {
      return {
        type: "wardrobe",
        imageUrl: await getImageUrlFromBase64(
          result.data.imageB64,
          req.user._id
        ),
      };
    } else {
      return {
        type: "wardrobe",
        imageUrl: null,
      };
    }
  } catch (error) {
    console.error("Error processing wardrobe item:", error);
    return {
      type: "wardrobe",
      imageUrl: null,
    };
  }
};
