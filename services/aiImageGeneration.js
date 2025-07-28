import { ai } from "../index.js";
import { getImageUrlFromBase64 } from "../services/base64ToImageUrl.js";

export const generateAIFashionSuggestions = async (
  occasion,
  numberOfImages,
  description = "",
  req
) => {
  try {
    const userBodyShape = req.user?.userBodyInfo?.bodyShape || "";
    const userUnderTone = req.user?.userBodyInfo?.undertone || "";

    // Set default height for women if not provided (average women's height: 5'4")
    const userHeight =
      req.user?.userBodyInfo?.height &&
      (req.user.userBodyInfo.height.feet > 0 ||
        req.user.userBodyInfo.height.inches > 0)
        ? req.user.userBodyInfo.height
        : { feet: 5, inches: 4 };

    // Build height description for the prompt
    const heightDescription = `${userHeight.feet}'${userHeight.inches}"`;

    // Build body shape styling guidelines
    const getBodyShapeGuidelines = (bodyShape) => {
      const guidelines = {
        pear: "Focus on balancing proportions by highlighting the upper body, using A-line silhouettes, and choosing tops that draw attention upward",
        apple:
          "Emphasize the legs and neckline, use empire waists, flowing fabrics, and avoid tight-fitting tops around the midsection",
        hourglass:
          "Highlight the natural waistline with fitted silhouettes, wrap styles, and tailored pieces that showcase the balanced proportions",
        rectangle:
          "Create curves and definition with layering, belted waists, peplum styles, and pieces that add volume to hips and bust",
        "inverted triangle":
          "Balance broad shoulders with wider bottom silhouettes, bootcut pants, A-line skirts, and softer shoulder lines",
      };
      return (
        guidelines[bodyShape.toLowerCase()] ||
        "Focus on creating flattering, well-fitted silhouettes that enhance natural proportions"
      );
    };

    // Build undertone color guidelines
    const getUndertoneColorGuidelines = (undertone) => {
      const colorGuidelines = {
        warm: "Use warm colors like coral, peach, golden yellow, warm reds, olive green, and cream. Avoid cool-toned colors like icy blues or stark whites",
        cool: "Use cool colors like navy blue, emerald green, royal purple, true red, and crisp whites. Avoid warm yellows, oranges, or golden tones",
        neutral:
          "Can wear both warm and cool colors effectively. Focus on colors that complement the occasion and outfit aesthetic",
      };
      return (
        colorGuidelines[undertone.toLowerCase()] ||
        "Choose colors that complement the skin tone and enhance the overall look"
      );
    };

    // Build description section for the prompt
    const descriptionSection =
      description && description.trim()
        ? `\n📝 ADDITIONAL STYLING REQUIREMENTS:\n- ${description.trim()}\n- Incorporate these specific preferences across all ${numberOfImages} outfits while maintaining variety\n- Balance user preferences with occasion appropriateness and outfit diversity\n- Let the description influence the overall aesthetic direction of all looks\n`
        : "";

    // Build personalization section
    const personalizationSection = `
🧍‍♀️ USER PERSONALIZATION (CRITICAL - Apply to ALL outfits):
${
  userBodyShape
    ? `- Body Shape: ${userBodyShape} - ${getBodyShapeGuidelines(
        userBodyShape
      )}`
    : ""
}
${
  userUnderTone
    ? `- Skin Undertone: ${userUnderTone} - ${getUndertoneColorGuidelines(
        userUnderTone
      )}`
    : ""
}
- Height: ${heightDescription} - Choose proportions and lengths that flatter this height
- All styling choices must consider these physical characteristics for maximum flattery
- Each outfit should be optimized for this specific body type and coloring
`;

    // Build model appearance section
    const modelAppearanceSection = `
👤 MODEL APPEARANCE REQUIREMENTS:
- Generate a photorealistic female model with these characteristics:
  ${
    userHeight
      ? `- Height: ${heightDescription} (adjust proportions accordingly)`
      : ""
  }
  ${
    userBodyShape
      ? `- Body shape: ${userBodyShape} body type with natural, realistic proportions`
      : ""
  }
  ${
    userUnderTone
      ? `- Skin tone: Natural ${userUnderTone} undertone complexion`
      : ""
  }
- Model should have a natural, approachable appearance
- Professional but relatable fashion model aesthetic
- Consistent model appearance across all ${numberOfImages} generated images
`;

    const aiPrompt = `
ROLE: You are a professional fashion stylist creating complete, full-body styled outfits for a personalized fashion editorial campaign, with expertise in both contemporary and traditional Indian ethnic wear.

OBJECTIVE:
Design ${numberOfImages} distinct cohesive outfits perfect for a **${occasion}** setting, specifically tailored for the user's body characteristics. Create complete looks styled on a model that represents the user's physical attributes.
${descriptionSection}
${personalizationSection}

✅ OUTFIT STRUCTURE (for each look):
${
  occasion &&
  (occasion.toLowerCase().includes("festival") ||
    occasion.toLowerCase().includes("wedding") ||
    occasion.toLowerCase().includes("diwali") ||
    occasion.toLowerCase().includes("holi") ||
    occasion.toLowerCase().includes("navratri") ||
    occasion.toLowerCase().includes("durga puja") ||
    occasion.toLowerCase().includes("karva chauth") ||
    occasion.toLowerCase().includes("indian") ||
    occasion.toLowerCase().includes("ethnic") ||
    occasion.toLowerCase().includes("traditional"))
    ? `
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
`
    : `
**For Contemporary/Western Occasions:**
- Include EXACTLY ONE complete outfit with:
  - One topwear (shirt, blouse, t-shirt, sweater, etc.)
  - One bottomwear (pants, skirt, shorts, etc.) OR a dress (if dress, no separate top/bottom needed)
  - One pair of shoes
  - 2–3 accessories (bag, hat, jewelry, belt, scarf – choose what fits the occasion)
`
}

🎨 STYLING FOCUS:
- Each outfit must be appropriate and stylish for **${occasion}**
- Make each look distinctly different in style, color palette, and approach
- Consider the formality level and aesthetic that suits this occasion
- Focus on creating cohesive, well-coordinated looks that FLATTER the user's specific body type
- Ensure variety between the ${numberOfImages} different outfits
${
  userBodyShape
    ? `- All silhouettes and fits must be optimized for ${userBodyShape} body shape`
    : ""
}
${
  userUnderTone
    ? `- All color choices must complement ${userUnderTone} undertones`
    : ""
}
${
  description && description.trim()
    ? `- Reflect the styling preferences: "${description.trim()}" in each of the ${numberOfImages} looks while maintaining distinct differences between them`
    : ""
}

${modelAppearanceSection}

🖼️ VISUAL OUTPUT FORMAT:
- **FULL-SIZE, HIGH-RESOLUTION IMAGES** - Generate complete, detailed fashion editorial images
- Each outfit styled on the personalized photorealistic human model
- **FULL-BODY MODEL SHOTS** - Show entire outfit from head to toe clearly
- Professional editorial photography look with studio-quality lighting
- Clean and neutral background (white, cream, or subtle gradient)
- Model should be positioned to showcase the complete outfit effectively
- **IMAGE DIMENSIONS**: Generate wide, full-resolution images that capture all styling details
- Model should consistently represent the user's physical characteristics across all images
- Show fabric textures, embellishments, and styling details clearly
${
  description && description.trim()
    ? `- Ensure the overall visual aesthetic aligns with: "${description.trim()}"`
    : ""
}

${
  occasion &&
  (occasion.toLowerCase().includes("festival") ||
    occasion.toLowerCase().includes("wedding") ||
    occasion.toLowerCase().includes("diwali") ||
    occasion.toLowerCase().includes("holi") ||
    occasion.toLowerCase().includes("navratri") ||
    occasion.toLowerCase().includes("durga puja") ||
    occasion.toLowerCase().includes("karva chauth") ||
    occasion.toLowerCase().includes("indian") ||
    occasion.toLowerCase().includes("ethnic") ||
    occasion.toLowerCase().includes("traditional"))
    ? `
**Additional Visual Requirements for Indian Ethnic Wear:**
- Showcase traditional draping techniques (saree pleats, dupatta styling)
- Highlight intricate embroidery, embellishments, and fabric details
- Include traditional jewelry styling and placement
- Show authentic color combinations and pattern mixing
- Capture the elegance and richness of Indian traditional wear
- Ensure cultural authenticity and respectful representation
`
    : ""
}

🚫 ABSOLUTE RULES:
- Generate exactly ${numberOfImages} distinct outfit looks
- **MANDATORY**: All images must be FULL-SIZE and HIGH-RESOLUTION
- No styling alternatives within each look
- No text overlays or descriptions on images
- Focus on showcasing realistic, wearable outfits
- Model must consistently reflect the specified physical characteristics
- All outfit choices must be flattering for the specified body type and coloring
- **Ensure complete outfit visibility** - no cropping of important styling elements
${
  description && description.trim()
    ? `- All outfits should harmonize with the user's style direction while being distinctly different from each other`
    : ""
}
${
  occasion &&
  (occasion.toLowerCase().includes("festival") ||
    occasion.toLowerCase().includes("wedding") ||
    occasion.toLowerCase().includes("diwali") ||
    occasion.toLowerCase().includes("holi") ||
    occasion.toLowerCase().includes("navratri") ||
    occasion.toLowerCase().includes("durga puja") ||
    occasion.toLowerCase().includes("karva chauth") ||
    occasion.toLowerCase().includes("indian") ||
    occasion.toLowerCase().includes("ethnic") ||
    occasion.toLowerCase().includes("traditional"))
    ? `
- Maintain cultural sensitivity and authenticity in ethnic wear representation
- Ensure traditional garments are styled correctly and respectfully
`
    : ""
}

✨ GOAL:
Deliver ${numberOfImages} distinct and editorial-quality complete outfits suitable for the **${occasion}** setting, each styled professionally on a female model who represents the user's physical characteristics${
      userBodyShape ? ` (${userBodyShape} body shape)` : ""
    }${
      userUnderTone ? ` with ${userUnderTone} undertones` : ""
    } at ${heightDescription} height${
      description && description.trim()
        ? `, with styling that reflects: "${description.trim()}"`
        : ""
    }. 

${
  occasion &&
  (occasion.toLowerCase().includes("festival") ||
    occasion.toLowerCase().includes("wedding") ||
    occasion.toLowerCase().includes("diwali") ||
    occasion.toLowerCase().includes("holi") ||
    occasion.toLowerCase().includes("navratri") ||
    occasion.toLowerCase().includes("durga puja") ||
    occasion.toLowerCase().includes("karva chauth") ||
    occasion.toLowerCase().includes("indian") ||
    occasion.toLowerCase().includes("ethnic") ||
    occasion.toLowerCase().includes("traditional"))
    ? `
**Special Focus**: Create authentic, beautiful Indian ethnic wear looks that celebrate traditional craftsmanship while ensuring modern styling sensibilities and perfect fit for the user's body type.
`
    : ""
}

**IMAGE QUALITY MANDATE**: All generated images must be full-resolution, professional-quality fashion editorial photographs that showcase every styling detail clearly and beautifully.
`;

    // const aiResponse = await ai.models.generateContent({
    //   model: "gemini-2.0-flash-preview-image-generation",
    //   contents: [{ text: aiPrompt }],
    //   config: {
    //     responseModalities: ["TEXT", "IMAGE"],
    //     numberOfImages: numberOfImages || 2,
    //   },
    // });
    const aiResponse = await ai.models.generateContent({
      model: "imagen-3", // Use the latest, most capable image model
      contents: [{ text: aiPrompt }],
      // The config object is removed as it's not a valid parameter
    });

    // Extract AI generated images
    const generatedImages = [];

    for (const part of aiResponse.candidates[0].content.parts) {
      if (part.inlineData) {
        const imagePart = part;
        if (imagePart && imagePart.inlineData?.data) {
          generatedImages.push({
            type: "ai_generated",
            imageUrl: await getImageUrlFromBase64(
              imagePart.inlineData.data,
              req.user._id
            ),
          });
        } else {
          console.error("No image data returned from AI model", { part });
          generatedImages.push({
            type: "ai_generated",
            imageUrl: null,
          });
        }
      }
    }

    return {
      generatedImages,
    };
  } catch (error) {
    console.error("Error generating AI fashion suggestions:", error);
    return {
      type: "ai_generated",
      imageB64: null,
    };
  }
};
