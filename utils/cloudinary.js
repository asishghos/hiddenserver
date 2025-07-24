import { v2 as cloudinary } from 'cloudinary';
import fs from 'fs';
import dotenv from 'dotenv';
dotenv.config();

cloudinary.config({
    cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
    api_key: process.env.CLOUDINARY_API_KEY,
    api_secret: process.env.CLOUDINARY_API_SECRET,
    secure: true,
});

const uploadOnCloudinary = async (localFilePath) => {
    try {
        if (!localFilePath) return null;

        const response = await cloudinary.uploader.upload(localFilePath, {
            resource_type: "auto",
        });

        fs.unlink(localFilePath, (err) => {
            if (err) {
                console.error("Error deleting local file:", err);
            } else {
                console.log("Local file deleted successfully");
            }
        });

        return response.secure_url || response.url;
    } catch (error) {
        console.error("Error uploading to Cloudinary:", error);
        fs.unlink(localFilePath, (err) => {
            if (err) {
                console.error("Error deleting local file:", err);
            }
        });
        return null;
    }
}

const deleteFromCloudinary = async (oldImageUrl) => {
    try {
        if (!oldImageUrl) return null;
        const publicId = oldImageUrl.split('/').pop().split('.')[0];
        const response = await cloudinary.uploader.destroy(publicId, {
            resource_type: 'image',
        });
        if (response.result === 'ok') {
            console.log("Image deleted successfully from Cloudinary");
        } else {
            console.error("Failed to delete image from Cloudinary:", response);
        }
    } catch (error) {
        console.error("Error deleting from Cloudinary:", error);
    }
}

export { uploadOnCloudinary, deleteFromCloudinary };