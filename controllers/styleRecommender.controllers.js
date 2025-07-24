import { uploadAndValidateWithCritique } from "../services/fashionValidator.js";
import { generateModelImage } from "../services/mannequinGenerator.js";
import { processWardrobeItemForOccasion } from "../services/aiImageFromWardrobeItem.js";
import { generateAIFashionSuggestions } from "../services/aiImageGeneration.js";
import { analyzeGeneratedAndBadImages } from "../services/imageAnalysisService.js";

// import { deleteFromCloudinary } from "../utils/cloudinary.js";
// Helper function to get reasons for bad items
// function getBadItemReasons(badItemImages, suitabilityDetails, imageUrls) {
//   if (!badItemImages?.length || !suitabilityDetails) return [];

//   const labels = ["Topwear", "Bottomwear", "Accessory", "Footwear"];

//   return badItemImages.map((badImageUrl) => {
//     // Find the index of this bad image in imageUrls
//     const imageIndex = imageUrls.findIndex((url) => url === badImageUrl);

//     if (imageIndex !== -1 && labels[imageIndex]) {
//       const label = labels[imageIndex];
//       const itemDetails = suitabilityDetails[label];

//       return {
//         imageUrl: badImageUrl,
//         itemType: label,
//         reason: itemDetails?.reasoning || "Not suitable for the occasion",
//       };
//     }

//     return {
//       imageUrl: badImageUrl,
//       itemType: "Unknown",
//       reason: "Not suitable for the occasion",
//     };
//   });
// }

export const styleRecommenderController = async (req, res) => {
  try {
    const imageFiles = req.files;
    const occasion = req.body.occasion;
    const description = req.body.description || ""; // Optional description for wardrobe item
    // console.log(description);
    if (!imageFiles?.length) {
      return res.status(400).json({ error: "No images provided" });
    }

    if (!occasion) {
      return res.status(400).json({ error: "Occasion is required" });
    }

    // Single merged function call for validation and critique
    const validationResult = await uploadAndValidateWithCritique(
      imageFiles,
      occasion,
      req
    );

    if (validationResult.error) {
      return res.status(400).json({ error: validationResult.error });
    }

    const {
      imageUrls,
      critique,
      isPerfectMatch,
      badItemImages,
      suitabilityDetails,
    } = validationResult;

    // let modelImage = null;
    // let wardrobeImageResponse = null;
    // let aiGeneratedImageResponse = null;

    // Always try to get wardrobe image first
    let wardrobeImageResponse = await processWardrobeItemForOccasion(
      req,
      occasion,
      description
    );
    // console.log(wardrobeImageResponse);
    // Generate content for all cases
    let modelImage = await generateModelImage(
      imageUrls,
      occasion,
      badItemImages,
      description,
      req
    );
    // console.log(modelImage);
    // Determine how many AI images to generate based on wardrobe availability
    const aiImageCount = wardrobeImageResponse?.data ? 1 : 2;

    // ispe badimages nhi ja rha hai ......................................................................
    let aiGeneratedImageResponse = await generateAIFashionSuggestions(
      occasion,
      aiImageCount,
      description,
      req
    );
    // console.log(aiGeneratedImageResponse);
    const results = [];

    if (wardrobeImageResponse?.imageUrl) {
      results.push(wardrobeImageResponse);
    }

    if (modelImage?.imageUrl) {
      results.push(modelImage);
    }

    if (aiGeneratedImageResponse?.generatedImages?.length) {
      aiGeneratedImageResponse.generatedImages.forEach((img) => {
        if (img?.imageUrl) results.push(img);
      });
    }

    // console.log(results);

    // Analyze generated images and bad images with AI
    let imageAnalysis = null;
    try {
      imageAnalysis = await analyzeGeneratedAndBadImages(
        results,
        badItemImages,
        occasion
      );
    } catch (error) {
      console.error("Error analyzing images:", error);
    }

    // console.log("image generate successfully");
    res.status(200).json({
      results,
      // badItemReasons: getBadItemReasons(
      //   badItemImages,
      //   suitabilityDetails,
      //   imageUrls
      // ),
      imageAnalysis,
    });

    // await Promise.all(imageUrls.map(deleteFromCloudinary(url => url)));
  } catch (error) {
    console.error("Error in styleRecommenderController:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
};
