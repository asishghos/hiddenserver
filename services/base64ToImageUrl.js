import { uploadOnCloudinary } from "../utils/cloudinary.js";
import fs from "fs";
import path from "path";

export const getImageUrlFromBase64 = async (imageB64, userId = "") => {
  let tempFilePath = null;
  try {
    if (!imageB64 || typeof imageB64 !== "string") {
      console.error("Invalid base64 input");
      return null;
    }

    const base64Data = imageB64.replace(/^data:image\/\w+;base64,/, "");
    const buffer = Buffer.from(base64Data, "base64");

    // Create temporary directory if needed
    const tempDir = "./temp";
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }

    // Create temporary file path
    tempFilePath = path.join(
      tempDir,
      `temp_${userId || "user"}_${Date.now()}.jpg`
    );
    fs.writeFileSync(tempFilePath, buffer);

    // Upload to Cloudinary
    const imageUrl = await uploadOnCloudinary(tempFilePath);

    if (!imageUrl) {
      console.error("Failed to upload image to Cloudinary");
      return null;
    }

    return imageUrl;
  } catch (error) {
    console.error("Error in getImageUrlFromBase64:", error);
    return null;
  } // Final cleanup block
finally {
  if (tempFilePath) {
    try {
      if (fs.existsSync(tempFilePath)) {
        fs.unlinkSync(tempFilePath);
        console.log("Local file deleted successfully");
      }
    } catch (cleanupError) {
      console.warn("Temp file cleanup skipped:", cleanupError.message);
    }
  }
}

};
