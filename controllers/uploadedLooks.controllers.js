import { UploadedLooks } from "../models/uploadedLooks.models.js";
import { validateChatbotImage } from "../services/validateChatbotImages.js";
import {
  uploadOnCloudinary,
  deleteFromCloudinary,
} from "../utils/cloudinary.js";
import { addToWardrobeFromUrl } from "../services/addToWardrobe.js";
import fs from "fs/promises";

// add to uploaded looks
export const addUploadedLook = async (req, res) => {
  const user = req.user;
  const imageFile = req.file;
  const { userQuery = "" } = req.body || {};

  if (!user || !imageFile) {
    return res.status(400).json({ message: "User or image file is missing" });
  }

  let imageUrl = null;

  try {
    // Read the image buffer once for all operations
    const imageBuffer = await fs.readFile(imageFile.path);

    // Step 1: Upload to Cloudinary
    imageUrl = await uploadOnCloudinary(imageFile.path);
    if (!imageUrl) {
      return res
        .status(500)
        .json({ message: "Error uploading image to Cloudinary" });
    }

    // Step 2: Validate image for fashion + full body
    const validationResult = await validateChatbotImage(imageUrl, userQuery);
    const { containsFullBodyHuman, generatedTitle, containsFashionItem } =
      validationResult;

    // Step 3: If not a full-body human, skip uploaded look
    if (!containsFullBodyHuman) {
      console.log(
        "Image does not contain a full-body human, skipping uploaded look"
      );
      await deleteFromCloudinary(imageUrl); // cleanup Cloudinary since we won't save to DB
      // await deleteLocalFile(imageFile.path); // cleanup local file

      return res.status(204).json({
        message: "Image skipped: no full-body human",
        data: null,
      });
    }

    // Step 4: Save to UploadedLooks
    const newLook = new UploadedLooks({
      userId: user._id,
      imageUrl,
      title: generatedTitle || "Your Uploaded Look",
    });

    await newLook.save();

    // Step 5: If it's a valid fashion item, add to wardrobe using URL and buffer
    if (containsFashionItem) {
      try {
        await addToWardrobeFromUrl(
          user._id,
          imageUrl,
          imageBuffer,
          imageFile.originalname
        );
        console.log("Successfully added to wardrobe");
      } catch (error) {
        console.error("Error adding to wardrobe:", error);
        // Continue even if wardrobe addition fails
      }
    }

    // Clean up local file immediately after all operations
    // await deleteLocalFile(imageFile.path);

    return res.status(201).json({
      message: "Look uploaded successfully",
      data: newLook,
    });
  } catch (error) {
    console.error("Error adding uploaded look:", error);

    // Cleanup on error
    if (imageUrl) {
      try {
        await deleteFromCloudinary(imageUrl);
      } catch (cleanupError) {
        console.error("Error cleaning up Cloudinary image:", cleanupError);
      }
    }

    // Always try to cleanup local file on error
    // try {
    //     await deleteLocalFile(imageFile.path);
    // } catch (cleanupError) {
    //     console.log("File cleanup error (file may already be deleted):", cleanupError);
    // }

    return res.status(500).json({ message: "Server error" });
  }
};

// get all uploaded looks
export const getUploadedLooks = async (req, res) => {
  try {
    const userId = req.user._id;

    const looks = await UploadedLooks.find({ userId }).sort({ createdAt: -1 });

    res.status(200).json({ looks });
  } catch (error) {
    console.error("Error fetching looks:", error);
    res
      .status(500)
      .json({ message: "Failed to fetch looks", error: error.message });
  }
};

// get a look by id
export const getLookById = async (req, res) => {
  try {
    const userId = req.user._id;
    const { lookId } = req.params;
    const look = await UploadedLooks.findOne({ _id: lookId, userId });
    if (!look) {
      return res
        .status(404)
        .json({ message: "Look not found or not authorized to access." });
    }
    res.status(200).json({ look });
  } catch (error) {
    console.error("Error fetching look:", error);
    res
      .status(500)
      .json({ message: "Failed to fetch look", error: error.message });
  }
};

// delete an uploaded look
export const deleteUploadedLook = async (req, res) => {
  try {
    const userId = req.user._id;
    const { lookId } = req.params;

    const look = await UploadedLooks.findOne({ _id: lookId, userId });

    if (!look) {
      return res
        .status(404)
        .json({ message: "Look not found or not authorized to delete." });
    }

    const imageUrl = look.imageUrl;

    const deleted = await UploadedLooks.findOneAndDelete({
      _id: lookId,
      userId,
    });

    if (imageUrl) {
      await deleteFromCloudinary(imageUrl);
    }

    if (!deleted) {
      return res
        .status(404)
        .json({ message: "Look not found or not authorized to delete." });
    }

    res.status(200).json({ message: "Look deleted successfully" });
  } catch (error) {
    console.error("Error deleting look:", error);
    res
      .status(500)
      .json({ message: "Failed to delete look", error: error.message });
  }
};
