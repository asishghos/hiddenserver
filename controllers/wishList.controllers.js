import { Wishlist } from "../models/wishList.models.js";

// Toggle wishlist item (add if not present, remove if exists)
export const toggleWishlistItem = async (req, res) => {
  const userId = req.user._id;

  if (!userId) {
    return res.status(401).json({ msg: "Unauthorized access to wishlist" });
  }
  try {
    const {
      productId,
      productTitle,
      productImage,
      price,
      originalPrice,
      discountPercent,
      platform,
      rating,
      productUrl,
    } = req.body;

    if (!productId || !productTitle || !price || !platform) {
      return res.status(400).json({ msg: "Missing required product fields" });
    }

    // Check if the item already exists
    const existingItem = await Wishlist.findOne({ userId, productUrl });

    if (existingItem) {
      // If exists, remove it
      await Wishlist.deleteOne({ _id: existingItem._id });
      return res.status(200).json({
        msg: "Product removed from wishlist",
        data: productId,
      });
    } else {
      // If not, add it
      const newItem = await Wishlist.create({
        userId,
        productId,
        productTitle,
        productImage,
        price,
        originalPrice,
        discountPercent,
        platform,
        rating,
        productUrl,
      });
      return res.status(201).json({
        msg: "Product added to wishlist",
        data: newItem,
      });
    }
  } catch (error) {
    console.error("Wishlist toggle error:", error.message);
    res.status(500).json({ msg: "Server error while toggling wishlist" });
  }
};

// Get all wishlist items for a user
export const getWishlistItems = async (req, res) => {
  const userId = req.user._id;

  if (!userId) {
    return res.status(401).json({ msg: "Unauthorized access to wishlist" });
  }

  try {
    const items = await Wishlist.find({ userId }).sort({ createdAt: -1 }); // newest first
    res.status(200).json({
      msg: "Wishlist items fetched successfully",
      count: items.length,
      data: items,
    });
  } catch (error) {
    console.error("Error fetching wishlist items:", error.message);
    res.status(500).json({ msg: "Server error while fetching wishlist items" });
  }
};
