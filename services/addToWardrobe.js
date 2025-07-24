import fs from 'fs/promises';
import { DigitalWardrobe } from '../models/digitalWardrobe.models.js';
import { uploadOnCloudinary } from '../utils/cloudinary.js';
import { extractClothingMetadata } from '../index.js';
import { generateImageHash } from '../controllers/digitalWardrobe.controllers.js';
import { User } from '../models/users.models.js';

export const addToWardrobe = async (userId, files) => {
  const user = await User.findById(userId);
  if (!user) throw new Error('User not found');

  let processedCount = 0;
  let skippedCount = 0;

  for (const file of files) {
    try {
      const imageBuffer = await fs.readFile(file.path);
      const imageHash = generateImageHash(imageBuffer);

      const existingWardrobe = await DigitalWardrobe.findOne({
        userId,
        'uploadedImages.imageHash': imageHash
      });

      if (existingWardrobe) {
        skippedCount++;
        console.log(`Image ${file.filename} already exists, skipping...`);
        continue;
      }

      const base64Image = imageBuffer.toString('base64');
      const metadata = await extractClothingMetadata(base64Image, file.mimetype);

      if (!metadata || !Array.isArray(metadata) || metadata.length === 0) {
        console.error(`Failed to extract metadata for ${file.filename}`);
        continue;
      }

      const imageUrl = await uploadOnCloudinary(file.path);
      if (!imageUrl) {
        console.error(`Failed to upload ${file.filename} to cloudinary`);
        continue;
      }

      const garments = metadata
        .filter(item => item?.itemName && item?.category && item?.color?.name && item?.color?.hex && item?.fabric)
        .map(item => ({
          itemName: item.itemName.trim(),
          category: item.category,
          color: {
            name: item.color.name.trim(),
            hex: item.color.hex.trim()
          },
          fabric: item.fabric,
          occasion: Array.isArray(item.occasion)
            ? item.occasion.slice(0, 3).map(occ => occ.trim())
            : [item.occasion?.trim()].filter(Boolean).slice(0, 3),
          season: Array.isArray(item.season)
            ? item.season.slice(0, 2)
            : [item.season].filter(Boolean).slice(0, 2)
        }));

      if (garments.length === 0) {
        console.error(`No valid garments found in ${file.filename}`);
        continue;
      }

      const imageEntry = {
        imageUrl,
        imageHash,
        garments,
        createdAt: new Date()
      };

      await DigitalWardrobe.findOneAndUpdate(
        { userId },
        { $push: { uploadedImages: imageEntry } },
        { upsert: true, new: true }
      );

      processedCount++;
      console.log(`Successfully processed ${file.filename} with ${garments.length} garments`);
    } catch (fileError) {
      console.error(`Error processing file ${file.filename}:`, fileError);
      continue;
    } finally {
      try {
        await fs.unlink(file.path);
      } catch (unlinkError) {
        console.error(`Failed to delete temporary file ${file.path}:`, unlinkError);
      }
    }
  }

  return { processed: processedCount, skipped: skippedCount, total: files.length };
};

// New version that works with imageUrl and imageBuffer directly
export const addToWardrobeFromUrl = async (userId, imageUrl, imageBuffer, filename = 'uploaded-image') => {
  const user = await User.findById(userId);
  if (!user) throw new Error('User not found');

  try {
    const imageHash = generateImageHash(imageBuffer);

    const existingWardrobe = await DigitalWardrobe.findOne({
      userId,
      'uploadedImages.imageHash': imageHash
    });

    if (existingWardrobe) {
      console.log(`Image ${filename} already exists, skipping...`);
      return { processed: 0, skipped: 1, total: 1 };
    }

    const base64Image = imageBuffer.toString('base64');
    const metadata = await extractClothingMetadata(base64Image, 'image/jpeg'); // You can pass the actual mimetype

    if (!metadata || !Array.isArray(metadata) || metadata.length === 0) {
      console.error(`Failed to extract metadata for ${filename}`);
      return { processed: 0, skipped: 0, total: 1 };
    }

    const garments = metadata
      .filter(item => item?.itemName && item?.category && item?.color?.name && item?.color?.hex && item?.fabric)
      .map(item => ({
        itemName: item.itemName.trim(),
        category: item.category,
        color: {
          name: item.color.name.trim(),
          hex: item.color.hex.trim()
        },
        fabric: item.fabric,
        occasion: Array.isArray(item.occasion)
          ? item.occasion.slice(0, 3).map(occ => occ.trim())
          : [item.occasion?.trim()].filter(Boolean).slice(0, 3),
        season: Array.isArray(item.season)
          ? item.season.slice(0, 2)
          : [item.season].filter(Boolean).slice(0, 2)
      }));

    if (garments.length === 0) {
      console.error(`No valid garments found in ${filename}`);
      return { processed: 0, skipped: 0, total: 1 };
    }

    const imageEntry = {
      imageUrl,
      imageHash,
      garments,
      createdAt: new Date()
    };

    await DigitalWardrobe.findOneAndUpdate(
      { userId },
      { $push: { uploadedImages: imageEntry } },
      { upsert: true, new: true }
    );

    console.log(`Successfully processed ${filename} with ${garments.length} garments`);
    return { processed: 1, skipped: 0, total: 1 };
  } catch (error) {
    console.error(`Error processing image ${filename}:`, error);
    return { processed: 0, skipped: 0, total: 1 };
  }
};