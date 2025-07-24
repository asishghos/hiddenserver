import { Router } from "express";
import multer from "multer";
import { createMask } from "../utils/createMask.js";
import { extractComplementaryClothMetaData } from "../index.js";
import { ai } from "../index.js";
import { Modality } from "@google/genai";
import mongoose from "mongoose";
import { DigitalWardrobe } from "../models/digitalWardrobe.models.js";
import {verifyJWT} from "../middleware/auth.middleware.js";

const router = Router();
const upload = multer(); // memory storage

router.post("/upload", verifyJWT, upload.single("image"), async (req, res) => {
    try {
        const { bodyPart: preservedType, occasion } = req.body;
        const file = req.file;
        const userId = req.user._id; // Assuming user is authenticated via middleware

        if (!file || !preservedType || !occasion) {
            return res.status(400).json({ message: "Missing required inputs" });
        }

        // Step 1: Mask + Base64
        const masked = await createMask(file.buffer);
        const base64Image = masked.buffer.toString("base64");

        // Step 2: Extract complementary item metadata using Gemini
        const complementaryMeta = await extractComplementaryClothMetaData(
            base64Image,
            occasion,
            preservedType,
            file.mimetype
        );

        if (!complementaryMeta) {
            return res.status(500).json({ message: "Failed to extract metadata" });
        }

        const {
            category,
            color,
            fabric,
            pattern,
            ai_tags = [],
            season = []
        } = complementaryMeta;

        // Step 3: Search MongoDB wardrobe for matching items
        const matchedItems = await DigitalWardrobe.aggregate([
            { $match: { userId: new mongoose.Types.ObjectId(userId) } },
            { $unwind: "$uploadedImages" },
            { $unwind: "$uploadedImages.garments" },
            {
                $match: {
                    $or: [
                        { "uploadedImages.garments.category": category },
                        {
                            $or: [
                                { "uploadedImages.garments.color": color },
                                { "uploadedImages.garments.fabric": fabric },
                                { "uploadedImages.garments.pattern": pattern },
                                { "uploadedImages.garments.ai_tags": { $in: [ai_tags] } },
                                { "uploadedImages.garments.season": { $in: [season] } }
                            ]
                        },
                        { "uploadedImages.garments.occasion": occasion } // Correct for occasion: [String]
                    ]
                }
            }
            ,
            {
                $project: {
                    imageUrl: "$uploadedImages.imageUrl",
                    garment: "$uploadedImages.garments"
                }
            }
        ]);

        // Step 4: Generate 2 AI outfit images with distinct styles
        const results = [];
        const previousOutfitDescriptions = [];

        for (let i = 0; i < 2; i++) {
            const prompt = `
You are a fashion stylist designing Look ${i + 1} of 2 for a ${occasion} occasion in , featuring an unchanged ${preservedType}. Avoid overlap with previous looks.

${previousOutfitDescriptions.length > 0
                    ? previousOutfitDescriptions.map((desc, idx) => `Look ${idx + 1}: ${desc}`).join("\n")
                    : ""}

Include:
- ${preservedType.includes("top") ? "One bottomwear" : preservedType.includes("bottom") ? "One topwear" : "Optional outerwear for dress"}
- 1 pair of shoes
- 2-3 accessories

Do not alter the uploaded garment. Show it on a photorealistic model in a clean, fashion-magazine style.

ONE outfit only.
      `.trim();

            const contents = [
                { text: prompt },
                {
                    inlineData: {
                        mimeType: masked.mimeType,
                        data: base64Image,
                    }
                }
            ];

            const response = await ai.models.generateContent({
                model: "gemini-2.0-flash-preview-image-generation",
                contents,
                config: {
                    responseModalities: [Modality.TEXT, Modality.IMAGE]
                }
            });

            const parts = response?.candidates?.[0]?.content?.parts;
            const textDescription = parts?.[0]?.text;
            const imagePart = parts?.find((p) => p?.inlineData?.data);

            previousOutfitDescriptions.push(textDescription);

            results.push(
                imagePart
                    ? {
                        style: `Look ${i + 1}`,
                        description: textDescription,
                        base64: imagePart.inlineData.data
                    }
                    : {
                        style: `Look ${i + 1}`,
                        error: "No image returned from Gemini"
                    }
            );
        }

        // Step 5: Final response
        return res.status(200).json({
            wardrobeMatches: matchedItems,
            aiStyledLooks: results
        });

    } catch (err) {
        console.error("Error in /upload:", err);
        res.status(500).json({ message: "Internal server error" });
    }
});

export default router;
