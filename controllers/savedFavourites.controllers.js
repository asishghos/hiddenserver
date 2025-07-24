import { SavedFavourites } from "../models/savedFavourites.models.js";
import { uploadOnCloudinary } from "../utils/cloudinary.js";
import { deleteFromCloudinary } from "../utils/cloudinary.js";

// add image to saved favourites
export const addToSavedFavourites = async (req, res) => {    
    try {
        const userId = req.user._id;
        if (!userId) {
            return res.status(400).json({ msg: "User unauthorized to access this feature" });
        }

        const { tag, occasion, description } = req.body;
        const file = req.files?.[0];

        if (!file || !tag || !occasion || !description) {
            return res.status(400).json({ msg: "All fields are required including image file" });
        }

        // Upload image file to Cloudinary
        const imageUrl = await uploadOnCloudinary(file.path);
        if (!imageUrl) {
            return res.status(500).json({ msg: "Error uploading image to Cloudinary from saved favourites" });
        }

        // Create a new saved favourite
        const savedFavouriteImage = await SavedFavourites.create({
            userId,
            imageUrl,
            tag,
            occasion,
            description
        });

        return res.status(201).json({ data: savedFavouriteImage, msg: "Image added to saved favourites successfully" });
    } catch (error) {
        console.log("Error while adding to saved favourites:", error);
        return res.status(500).json({ msg: "Error while adding to saved favourites" });
    }
};

export const getSavedFavourites = async (req, res) => {
    try {
        const userId = req.user._id;
        if (!userId) {
            return res.status(400).json({ msg: "User unauthorized to access this feature" });
        }

        const savedFavourites = await SavedFavourites.find({ userId });

        if (!savedFavourites || savedFavourites.length === 0) {
            return res.status(404).json({ msg: "No saved favourites found" });
        }

        return res.status(200).json({ msg: "Favourites fetched successfully", count: savedFavourites.length, data: savedFavourites });
    } catch (error) {
        console.log("Error while fetching saved favourites:", error);
        return res.status(500).json({ msg: "Error while fetching saved favourites" });
    }
};

export const deleteSavedFavourite = async (req, res) => {
    try {
        const userId = req.user._id;
        const { favouriteId } = req.params;

        if (!userId) {
            return res.status(400).json({ msg: "User unauthorized to access this feature" });
        }

        if (!favouriteId) {
            return res.status(400).json({ msg: "Favourite ID is required" });
        }

        const savedFavourite = await SavedFavourites.findOne({ _id: favouriteId, userId });
        if (!savedFavourite) {
            return res.status(404).json({ msg: "Saved favourite not found" });
        }

        await deleteFromCloudinary(savedFavourite.imageUrl);
        await SavedFavourites.deleteOne({ _id: favouriteId });

        return res.status(200).json({ msg: "Saved favourite deleted successfully" });
    } catch (error) {
        console.log("Error while deleting saved favourite:", error);
        return res.status(500).json({ msg: "Error while deleting saved favourite" });
    }
};