import { extractClothingMetadata } from "../index.js";
import { DigitalWardrobe } from "../models/digitalWardrobe.models.js";
import {  addToWardrobe } from '../services/addToWardrobe.js';
import { User } from "../models/users.models.js";
import fs from "fs/promises";
import crypto from "crypto";
import {
  uploadOnCloudinary,
  deleteFromCloudinary,
} from "../utils/cloudinary.js";

const ALL_CATEGORIES = [
  "Tops",
  "Bottoms",
  "Ethnic",
  "Dresses",
  "co-ord set",
  "Swimwear",
  "Footwear",
  "Accessories",
];

export const getCategoryCounts = async (req, res) => {
  try {
    const userId = req.user._id;

    const result = await DigitalWardrobe.aggregate([
      { $match: { userId } },
      { $unwind: "$uploadedImages" },
      { $unwind: "$uploadedImages.garments" },
      {
        $group: {
          _id: "$uploadedImages.garments.category",
          count: { $sum: 1 },
        },
      },
      {
        $project: {
          _id: 0,
          category: "$_id",
          count: 1,
        },
      },
    ]);

    // Convert result to a map
    const countsMap = result.reduce((acc, curr) => {
      acc[curr.category] = curr.count;
      return acc;
    }, {});

    // Ensure all categories are included, defaulting to 0
    const counts = {};
    for (const category of ALL_CATEGORIES) {
      counts[category] = countsMap[category] || 0;
    }

    return res.status(200).json({
      message: "Category counts fetched successfully",
      counts,
    });
  } catch (error) {
    console.error("Error fetching category counts:", error);
    return res.status(500).json({
      message: "Failed to fetch category counts",
      error: process.env.NODE_ENV === "development" ? error.message : undefined,
    });
  }
};

export function generateImageHash(buffer) {
  return crypto.createHash("sha256").update(buffer).digest("hex");
}

// to add items into the wardrobe
export const addGarmentToDigitalWardrobe = async (req, res) => {
  try {
    const userId = req.user._id;
    const files = req.files || (req.file ? [req.file] : []);

    if (!files.length) {
      return res.status(400).json({ message: "No file uploaded" });
    }

    const { processed, skipped, total } = await  addToWardrobe(userId, files);

    return res.status(200).json({
      message: `Processing complete. ${processed} images added, ${skipped} skipped (duplicates)`,
      processed,
      skipped,
      total
    });

  } catch (error) {
    console.error("Error adding garment to wardrobe:", error);
    res.status(500).json({
      message: "Internal server error",
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};
// export const addGarmentToDigitalWardrobe = async (req, res) => {
//   try {
//     const userId = req.user._id;
//     const files = req.files || (req.file ? [req.file] : []);

//     const user = await User.findById(userId);
//     if (!user) {
//       return res.status(404).json({ message: "User not found" });
//     }

//     if (!files.length) {
//       return res.status(400).json({ message: "No file uploaded" });
//     }

//     let processedCount = 0;
//     let skippedCount = 0;

//     for (const file of files) {
//       try {
//         const imageBuffer = await fs.readFile(file.path);
//         const imageHash = generateImageHash(imageBuffer);

//         const existingWardrobe = await DigitalWardrobe.findOne({
//           userId,
//           "uploadedImages.imageHash": imageHash,
//         });

//         if (existingWardrobe) {
//           skippedCount++;
//           console.log(`Image ${file.filename} already exists, skipping...`);
//           continue;
//         }

//         const base64Image = imageBuffer.toString("base64");
//         const metadata = await extractClothingMetadata(
//           base64Image,
//           file.mimetype
//         );

//         if (!metadata || !Array.isArray(metadata) || metadata.length === 0) {
//           console.error(`Failed to extract metadata for ${file.filename}`);
//           continue;
//         }

//         const imageUrl = await uploadOnCloudinary(file.path);
//         if (!imageUrl) {
//           console.error(`Failed to upload ${file.filename} to cloudinary`);
//           continue;
//         }

//         const garments = metadata
//           .filter(
//             (item) =>
//               item &&
//               item.itemName &&
//               item.category &&
//               item.color?.name &&
//               item.color?.hex &&
//               item.fabric
//           )
//           .map((item) => ({
//             itemName: item.itemName.trim(),
//             category: item.category,
//             color: {
//               name: item.color.name.trim(),
//               hex: item.color.hex.trim(),
//             },
//             fabric: item.fabric,
//             occasion: Array.isArray(item.occasion)
//               ? item.occasion.slice(0, 3).map((occ) => occ.trim())
//               : [item.occasion?.trim()].filter(Boolean).slice(0, 3),
//             season: Array.isArray(item.season)
//               ? item.season.slice(0, 2)
//               : [item.season].filter(Boolean).slice(0, 2),
//           }));

//         if (garments.length === 0) {
//           console.error(`No valid garments found in ${file.filename}`);
//           continue;
//         }

//         const imageEntry = {
//           imageUrl,
//           imageHash,
//           garments,
//           createdAt: new Date(),
//         };

//         await DigitalWardrobe.findOneAndUpdate(
//           { userId },
//           { $push: { uploadedImages: imageEntry } },
//           { upsert: true, new: true }
//         );

//         processedCount++;
//         console.log(
//           `Successfully processed ${file.filename} with ${garments.length} garments`
//         );
//       } catch (fileError) {
//         console.error(`Error processing file ${file.filename}:`, fileError);
//         continue;
//       } finally {
//         try {
//           await fs.unlink(file.path);
//         } catch (unlinkError) {
//           console.error(
//             `Failed to delete temporary file ${file.path}:`,
//             unlinkError
//           );
//         }
//       }
//     }

//     const responseMessage = `Processing complete. ${processedCount} images added, ${skippedCount} images skipped (duplicates)`;

//     return res.status(200).json({
//       message: responseMessage,
//       processed: processedCount,
//       skipped: skippedCount,
//       total: files.length,
//     });
//   } catch (error) {
//     console.error("Error adding garment to wardrobe:", error);
//     res.status(500).json({
//       message: "Internal server error",
//       error: process.env.NODE_ENV === "development" ? error.message : undefined,
//     });
//   }
// };

export const addGarmentToDigitalWardrobeByCategory = async (req, res) => {
  try {
    const userId = req.user._id;
    const { category } = req.query;
    const files = req.files || (req.file ? [req.file] : []);

    if (!userId)
      return res.status(401).json({ message: "Unauthorized access" });
    if (!category)
      return res
        .status(400)
        .json({ message: "Category is required as query param" });
    if (!files.length)
      return res.status(400).json({ message: "No files uploaded" });

    const user = await User.findById(userId);
    if (!user) return res.status(404).json({ message: "User not found" });

    const mismatched = [];

    for (const file of files) {
      let imageUrl = null;
      try {
        const imageBuffer = await fs.readFile(file.path);
        const imageHash = generateImageHash(imageBuffer);

        const existing = await DigitalWardrobe.findOne({
          userId,
          "uploadedImages.imageHash": imageHash,
        });

        if (existing) {
          console.log("Duplicate image");
          continue;
        }

        const base64Image = imageBuffer.toString("base64");
        const metadata = await extractClothingMetadata(
          base64Image,
          file.mimetype
        );

        if (!metadata || !Array.isArray(metadata) || metadata.length === 0) {
          console.log("Metadata extraction failed");
          continue;
        }

        // Upload to Cloudinary before checking mismatch (since you want to show URL)
        imageUrl = await uploadOnCloudinary(file.path);
        if (!imageUrl) {
          console.log("Upload to Cloudinary failed!");
          continue;
        }

        const validGarments = metadata.filter((g) => g.category === category);

        if (validGarments.length === 0) {
          mismatched.push({
            filename: file.originalname,
            imageUrl, // So frontend can show this in the alert
            reason: `No garment matched category "${category}"`,
            suggestedCategories: [...new Set(metadata.map((g) => g.category))],
          });
          continue;
        }

        const garments = validGarments.map((item) => ({
          itemName: item.itemName.trim(),
          category: item.category,
          color: {
            name: item.color.name.trim(),
            hex: item.color.hex.trim(),
          },
          fabric: item.fabric,
          occasion: Array.isArray(item.occasion)
            ? item.occasion.slice(0, 3).map((occ) => occ.trim())
            : [item.occasion?.trim()].filter(Boolean).slice(0, 3),
          season: Array.isArray(item.season)
            ? item.season.slice(0, 2)
            : [item.season].filter(Boolean).slice(0, 2),
        }));

        if (garments.length === 0) {
          console.log("No valid garments after filtering");
          continue;
        }

        const imageEntry = {
          imageUrl,
          imageHash,
          garments,
          createdAt: new Date(),
        };

        await DigitalWardrobe.findOneAndUpdate(
          { userId },
          { $push: { uploadedImages: imageEntry } },
          { upsert: true, new: true }
        );
      } catch (err) {
        console.error(`Error processing ${file.originalname}:`, err);
      } finally {
        try {
          await fs.unlink(file.path);
        } catch (err) {
          console.warn("Failed to delete temp file:", file.path);
        }
      }
    }

    return res.status(200).json({
      message: "Upload complete",
      mismatched,
    });
  } catch (err) {
    console.error("Error in addGarmentToDigitalWardrobeByCategory:", err);
    return res.status(500).json({
      message: "Internal server error",
      error: process.env.NODE_ENV === "development" ? err.message : undefined,
    });
  }
};

export const forceUploadMismatchedImages = async (req, res) => {
  try {
    const userId = req.user._id;
    const { images } = req.body;

    if (!userId)
      return res.status(401).json({ message: "Unauthorized access" });
    if (!Array.isArray(images) || images.length === 0) {
      return res
        .status(400)
        .json({ message: "No images provided to force-upload" });
    }

    const user = await User.findById(userId);
    if (!user) return res.status(404).json({ message: "User not found" });

    for (const image of images) {
      const { imageUrl, imageHash, metadata } = image;

      if (
        !imageUrl ||
        !imageHash ||
        !Array.isArray(metadata) ||
        metadata.length === 0
      ) {
        if (imageUrl) await deleteFromCloudinary(imageUrl);
        console.log("Invalid metadata or image info (or missing)");
        continue;
      }

      const existing = await DigitalWardrobe.findOne({
        userId,
        "uploadedImages.imageHash": imageHash,
      });

      if (existing) {
        if (imageUrl) await deleteFromCloudinary(imageUrl);
        console.log("Duplicate image");
        continue;
      }

      const garments = metadata
        .filter(
          (item) =>
            item &&
            item.itemName &&
            item.category &&
            item.color?.name &&
            item.color?.hex &&
            item.fabric
        )
        .map((item) => ({
          itemName: item.itemName.trim(),
          category: item.category,
          color: {
            name: item.color.name.trim(),
            hex: item.color.hex.trim(),
          },
          fabric: item.fabric,
          occasion: Array.isArray(item.occasion)
            ? item.occasion.slice(0, 3).map((occ) => occ.trim())
            : [item.occasion?.trim()].filter(Boolean).slice(0, 3),
          season: Array.isArray(item.season)
            ? item.season.slice(0, 2)
            : [item.season].filter(Boolean).slice(0, 2),
        }));

      if (garments.length === 0) {
        if (imageUrl) await deleteFromCloudinary(imageUrl);
        console.log("No valid garments found");
        continue;
      }

      const imageEntry = {
        imageUrl,
        imageHash,
        garments,
        createdAt: new Date(),
      };

      await DigitalWardrobe.findOneAndUpdate(
        { userId },
        { $push: { uploadedImages: imageEntry } },
        { upsert: true, new: true }
      );
    }

    return res.status(200).json({
      message: "Force upload complete",
    });
  } catch (err) {
    console.error("Error in forceUploadMismatchedImages:", err);
    return res.status(500).json({
      message: "Internal server error",
      error: process.env.NODE_ENV === "development" ? err.message : undefined,
    });
  }
};

export const updateGarment = async (req, res) => {
  try {
    const userId = req.user._id;
    const { garmentId } = req.params;
    const updatedFields = req.body;

    console.log(updatedFields);

    const wardrobe = await DigitalWardrobe.findOne({ userId });
    if (!wardrobe)
      return res.status(404).json({ message: "Wardrobe not found" });

    let garmentToUpdate = null;
    let parentImage = null;

    // Find the garment by its ID inside uploadedImages
    for (const image of wardrobe.uploadedImages) {
      const garment = image.garments.id(garmentId);
      if (garment) {
        garmentToUpdate = garment;
        parentImage = image;
        break;
      }
    }

    if (!garmentToUpdate)
      return res.status(404).json({ message: "Garment not found" });

    // Apply updates
    Object.assign(garmentToUpdate, updatedFields);
    await wardrobe.save();

    return res.status(200).json({
      message: "Garment updated successfully",
      updated: {
        ...garmentToUpdate.toObject(),
        imageUrl: parentImage?.imageUrl,
        createdAt: parentImage?.createdAt,
      },
    });
  } catch (err) {
    console.error("Update garment error:", err);
    return res
      .status(500)
      .json({ message: "Failed to update garment", error: err.message });
  }
};

export const deleteGarment = async (req, res) => {
  try {
    const userId = req.user._id;
    const { garmentId } = req.params;

    const wardrobe = await DigitalWardrobe.findOne({ userId });
    if (!wardrobe)
      return res.status(404).json({ message: "Wardrobe not found" });

    // Find the image containing this garment
    const targetImage = wardrobe.uploadedImages.find((image) =>
      image.garments.some((g) => g._id.toString() === garmentId)
    );

    if (!targetImage) {
      return res.status(404).json({ message: "Garment not found in wardrobe" });
    }

    // Remove garment from image
    const updatedGarments = targetImage.garments.filter(
      (g) => g._id.toString() !== garmentId
    );

    // If no garments left, delete entire image
    if (updatedGarments.length === 0) {
      await deleteFromCloudinary(targetImage.imageUrl);

      await DigitalWardrobe.findOneAndUpdate(
        { userId },
        { $pull: { uploadedImages: { _id: targetImage._id } } }
      );
    } else {
      // Replace garments in the matched image with the updated array
      await DigitalWardrobe.updateOne(
        { userId, "uploadedImages._id": targetImage._id },
        { $set: { "uploadedImages.$.garments": updatedGarments } }
      );
    }

    return res.status(200).json({ message: "Garment deleted successfully" });
  } catch (err) {
    console.error("Error deleting garment:", err);
    return res
      .status(500)
      .json({ message: "Internal server error", error: err.message });
  }
};

export const getGarmentsByCategory = async (req, res) => {
  try {
    const { category } = req.query;
    const userId = req.user._id;

    if (!userId) {
      return res
        .status(401)
        .json({ message: "Unauthorized access to wardrobe" });
    }

    if (!category) {
      return res
        .status(400)
        .json({ message: "Category is required as a query param" });
    }

    const wardrobe = await DigitalWardrobe.findOne({ userId });

    if (!wardrobe || wardrobe.uploadedImages.length === 0) {
      return res.status(404).json({ message: "No garments found in wardrobe" });
    }

    const sortedImages = [...wardrobe.uploadedImages].sort(
      (a, b) => new Date(b.createdAt) - new Date(a.createdAt)
    );

    if (category === "Recent") {
      const allGarments = sortedImages.flatMap((image) =>
        image.garments.map((garment) => ({
          imageId: image._id,
          garmentId: garment._id,
          itemName: garment.itemName,
          imageUrl: image.imageUrl,
          createdAt: image.createdAt,
        }))
      );

      return res.status(200).json({
        message: "All garments fetched successfully",
        results: allGarments,
      });
    }

    const result = [];

    for (const image of sortedImages) {
      for (const garment of image.garments) {
        if (garment.category === category) {
          result.push({
            imageId: image._id,
            garmentId: garment._id,
            itemName: garment.itemName,
            imageUrl: image.imageUrl,
            createdAt: image.createdAt,
          });
        }
      }
    }

    return res.status(200).json({
      message: `Garments fetched for category: ${category}`,
      results: result,
    });
  } catch (err) {
    console.error("Error in getGarmentsByCategory:", err);
    return res.status(500).json({
      message: "Failed to fetch garments by category",
      error: process.env.NODE_ENV === "development" ? err.message : undefined,
    });
  }
};

export const filterGarments = async (req, res) => {
  try {
    const userId = req.user._id;
    if (!userId) {
      return res.status(401).json({ message: "Unauthorized access" });
    }

    const { category, color, fabric, occasion, season } = req.query;

    const categoryFilter = category ? category.split(",") : null;
    const colorFilter = color ? color.split(",") : null;
    const fabricFilter = fabric ? fabric.split(",") : null;
    const occasionFilter = occasion ? occasion.split(",") : null;
    const seasonFilter = season ? season.split(",") : null;

    const wardrobe = await DigitalWardrobe.findOne({ userId });
    if (!wardrobe || wardrobe.uploadedImages.length === 0) {
      return res.status(404).json({ message: "No garments found in wardrobe" });
    }

    const filtered = [];

    for (const image of wardrobe.uploadedImages) {
      for (const garment of image.garments) {
        const isMatch =
          (!categoryFilter || categoryFilter.includes(garment.category)) &&
          (!fabricFilter || fabricFilter.includes(garment.fabric)) &&
          (!colorFilter || colorFilter.includes(garment.color?.name)) &&
          (!occasionFilter ||
            garment.occasion?.some((o) => occasionFilter.includes(o))) &&
          (!seasonFilter ||
            garment.season?.some((s) => seasonFilter.includes(s)));

        if (isMatch) {
          filtered.push({
            imageId: image._id,
            garmentId: garment._id,
            itemName: garment.itemName,
            imageUrl: image.imageUrl,
            createdAt: image.createdAt,
            // imageId: image._id,
            // garmentId: garment._id,
            // itemName: garment.itemName,
            // imageUrl: image.imageUrl,
            // createdAt: image.createdAt,
            // category: garment.category,
            // color: garment.color,
            // fabric: garment.fabric,
            // occasion: garment.occasion,
            // season: garment.season
          });
        }
      }
    }

    return res.status(200).json({
      message: "Filtered garments retrieved successfully",
      results: filtered,
    });
  } catch (err) {
    console.error("Error in filterGarments:", err);
    return res.status(500).json({
      message: "Failed to filter garments",
      error: process.env.NODE_ENV === "development" ? err.message : undefined,
    });
  }
};
// export const getGarmentsByFabric = async (req, res) => {
//   try {
//     const { fabric } = req.query;
//     const userId = req.user._id;

//     if (!fabric)
//       return res
//         .status(400)
//         .json({ message: "Fabric is required as query param" });

//     const wardrobe = await DigitalWardrobe.findOne({ userId });

//     if (!wardrobe || wardrobe.uploadedImages.length === 0)
//       return res.status(404).json({ message: "No garments found in wardrobe" });

//     const result = wardrobe.uploadedImages
//       .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
//       .flatMap((image) =>
//         image.garments
//           .filter((g) => g.fabric === fabric)
//           .map((g) => ({
//             imageId: image._id,
//             itemName: g.itemName,
//             garmentId: garment._id,
//             imageUrl: image.imageUrl,
//             createdAt: image.createdAt,
//           }))
//       );

//     res
//       .status(200)
//       .json({ message: `Garments with fabric: ${fabric}`, results: result });
//   } catch (err) {
//     console.error("Error fetching by fabric:", err);
//     res.status(500).json({ message: "Failed to fetch garments by fabric" });
//   }
// };

// export const getGarmentsByOccasion = async (req, res) => {
//   try {
//     const { occasion } = req.query;
//     const userId = req.user._id;

//     if (!occasion)
//       return res
//         .status(400)
//         .json({ message: "Occasion is required as query param" });

//     const wardrobe = await DigitalWardrobe.findOne({ userId });

//     if (!wardrobe || wardrobe.uploadedImages.length === 0)
//       return res.status(404).json({ message: "No garments found in wardrobe" });

//     const result = wardrobe.uploadedImages
//       .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
//       .flatMap((image) =>
//         image.garments
//           .filter((g) => g.occasion.includes(occasion))
//           .map((g) => ({
//             imageId: image._id,
//             itemName: g.itemName,
//             garmentId: garment._id,
//             imageUrl: image.imageUrl,
//             createdAt: image.createdAt,
//           }))
//       );

//     res.status(200).json({
//       message: `Garments with occasion: ${occasion}`,
//       results: result,
//     });
//   } catch (err) {
//     console.error("Error fetching by occasion:", err);
//     res.status(500).json({ message: "Failed to fetch garments by occasion" });
//   }
// };

// export const getGarmentsBySeason = async (req, res) => {
//   try {
//     const { season } = req.query;
//     const userId = req.user._id;

//     if (!season)
//       return res
//         .status(400)
//         .json({ message: "Season is required as query param" });

//     const wardrobe = await DigitalWardrobe.findOne({ userId });

//     if (!wardrobe || wardrobe.uploadedImages.length === 0)
//       return res.status(404).json({ message: "No garments found in wardrobe" });

//     const result = wardrobe.uploadedImages
//       .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
//       .flatMap((image) =>
//         image.garments
//           .filter((g) => g.season.includes(season))
//           .map((g) => ({
//             imageId: image._id,
//             itemName: g.itemName,
//             garmentId: garment._id,
//             imageUrl: image.imageUrl,
//             createdAt: image.createdAt,
//           }))
//       );

//     res
//       .status(200)
//       .json({ message: `Garments with season: ${season}`, results: result });
//   } catch (err) {
//     console.error("Error fetching by season:", err);
//     res.status(500).json({ message: "Failed to fetch garments by season" });
//   }
// };

// export const getGarmentsByColor = async (req, res) => {
//   try {
//     const { color } = req.query;
//     const userId = req.user._id;

//     if (!color)
//       return res
//         .status(400)
//         .json({ message: "Color is required as query param" });

//     const wardrobe = await DigitalWardrobe.findOne({ userId });

//     if (!wardrobe || wardrobe.uploadedImages.length === 0)
//       return res.status(404).json({ message: "No garments found in wardrobe" });

//     const result = wardrobe.uploadedImages
//       .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
//       .flatMap((image) =>
//         image.garments
//           .filter((g) => g.color.name === color)
//           .map((g) => ({
//             imageId: image._id,
//             itemName: g.itemName,
//             garmentId: garment._id,
//             imageUrl: image.imageUrl,
//             createdAt: image.createdAt,
//           }))
//       );

//     res
//       .status(200)
//       .json({ message: `Garments with color: ${color}`, results: result });
//   } catch (err) {
//     console.error("Error fetching by color:", err);
//     res.status(500).json({ message: "Failed to fetch garments by color" });
//   }
// };

export const getGarmentDetails = async (req, res) => {
  const { garmentId } = req.params;
  const userId = req.user._id;
  if (!userId) {
    return res.status(401).json({ message: "Unauthorized access to wardrobe" });
  }

  try {
    const wardrobe = await DigitalWardrobe.findOne({ userId });
    if (!wardrobe || wardrobe.uploadedImages.length === 0) {
      return res.status(404).json({ message: "No garments found in wardrobe" });
    }

    const garmentDetails = wardrobe.uploadedImages.flatMap((image) =>
      image.garments
        .filter((g) => g._id.toString() === garmentId)
        .map((g) => ({
          imageId: image._id,
          garmentId: g._id,
          itemName: g.itemName,
          category: g.category,
          color: g.color,
          fabric: g.fabric,
          occasion: g.occasion,
          season: g.season,
          imageUrl: image.imageUrl,
          createdAt: image.createdAt,
        }))
    );

    if (garmentDetails.length === 0) {
      return res.status(404).json({ message: "Garment not found" });
    }

    return res.status(200).json({
      message: "Garment details fetched successfully",
      garmentDetails: garmentDetails[0],
    });
  } catch (err) {
    console.error("Error fetching garment details:", err);
    return res.status(500).json({
      message: "Failed to fetch garment details",
      error: process.env.NODE_ENV === "development" ? err.message : undefined,
    });
  }
};
