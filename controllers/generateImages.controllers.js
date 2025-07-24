import { processWardrobeItemForOccasion } from "../services/aiImageFromWardrobeItem.js";
import { generateAIFashionSuggestions } from "../services/aiImageGeneration.js";
import { analyzeGeneratedAndBadImages } from "../services/imageAnalysisService.js";

export const generateImageForOccasion = async (req, res) => {
  try {
    const { occasion, description = "" } = req.body;
    console.log(description);
    if (!occasion) {
      return res.status(400).json({ error: "Missing occasion" });
    }

    const results = [];
    let wardrobeImageGenerated = false;
    let wardrobeItemsAvailable = 0;

    // Process wardrobe items using the service
    const wardrobeResult = await processWardrobeItemForOccasion(
      req,
      occasion,
      description
    ); // Add description parameter

    // console.log(wardrobeResult);
    if (wardrobeResult?.imageUrl) {
      results.push(wardrobeResult);
    }

    const aiImageCount = wardrobeImageGenerated ? 2 : 3;

    // Generate AI fashion suggestions using the service
    const aiGeneratedImageResponse = await generateAIFashionSuggestions(
      occasion,
      aiImageCount,
      description,
      req
    );

    // console.log(aiGeneratedImageResponse);

    if (aiGeneratedImageResponse?.generatedImages?.length) {
      aiGeneratedImageResponse.generatedImages.forEach((img) => {
        if (img?.imageUrl) results.push(img);
      });
    }

    // console.log(results);

    // Analyze generated images and bad images with AI
    let imageAnalysis = null;
    try {
      imageAnalysis = await analyzeGeneratedAndBadImages(results, [], occasion);
    } catch (error) {
      console.error("Error analyzing images:", error);
    }

    return res.status(200).json({
      results,
      imageAnalysis
    });
  } catch (error) {
    console.error("Error in generateImageForOccasion:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
};
