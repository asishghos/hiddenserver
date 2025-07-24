import { ZuriMagazine } from "../models/zuriMagazine.models.js";
import { uploadOnCloudinary, deleteFromCloudinary } from "../utils/cloudinary.js";
import { Bookmark } from "../models/bookMarkArticles.models.js";

// Create a new article
export const addArticle = async (req, res) => {
  try {
    const { authorName, category, title, content, subTitle, tags } = req.body;
    const files = req.files;

    if (!authorName || !category || !title || !content) {
      return res.status(400).json({ msg: "Required fields are missing" });
    }

    let authorProfilePic = undefined;
    let bannerImage = undefined;

    // Handle author profile pic upload
    if (files && files.authorProfilePic && files.authorProfilePic[0]) {
      const result = await uploadOnCloudinary(files.authorProfilePic[0].path);
      authorProfilePic = result;
    }

    // Handle banner image upload
    if (files && files.bannerImage && files.bannerImage[0]) {
      const result = await uploadOnCloudinary(files.bannerImage[0].path);
      bannerImage = result;
    }

    // Parse tags from JSON string if it exists
    let parsedTags = [];
    if (tags) {
      try {
        parsedTags = JSON.parse(tags);
      } catch (error) {
        // Fallback to comma-separated parsing
        parsedTags = tags.split(",").map((tag) => tag.trim());
      }
    }

    const newArticle = await ZuriMagazine.create({
      authorName,
      authorProfilePic,
      category: category.toLowerCase(),
      title,
      content,
      subTitle,
      bannerImage,
      tags: parsedTags
    });

    return res.status(201).json({ msg: "Article created successfully", data: newArticle });
  } catch (error) {
    console.error("Error adding article:", error);
    return res.status(500).json({ msg: "Internal Server Error" });
  }
};

// Update an existing article (excluding author details)
export const updateArticle = async (req, res) => {
  try {
    const { id } = req.params;
    const { category, title, content, subTitle, tags } = req.body;
    const imageFile = req.file;

    const existing = await ZuriMagazine.findById(id);
    if (!existing) {
      return res.status(404).json({ msg: "Article not found" });
    }

    // Handle banner image update
    if (imageFile) {
      // Delete old banner image if exists
      if (existing.bannerImage) {
        await deleteFromCloudinary(existing.bannerImage);
      }
      const result = await uploadOnCloudinary(imageFile.path);
      existing.bannerImage = result;
    }

    // Update only allowed fields (excluding authorName and authorProfilePic)
    if (category) existing.category = category.toLowerCase();
    if (title) existing.title = title;
    if (content) existing.content = content;
    if (subTitle !== undefined) existing.subTitle = subTitle; // Allow empty string
    
    if (tags) {
      let parsedTags = [];
      try {
        parsedTags = JSON.parse(tags);
      } catch (error) {
        // Fallback to comma-separated parsing
        parsedTags = tags.split(",").map((tag) => tag.trim());
      }
      existing.tags = parsedTags;
    }

    await existing.save();

    return res.status(200).json({ msg: "Article updated successfully", data: existing });
  } catch (error) {
    console.error("Error updating article:", error);
    return res.status(500).json({ msg: "Internal Server Error" });
  }
};

// Delete an article
export const deleteArticle = async (req, res) => {
  try {
    const { id } = req.params;

    const article = await ZuriMagazine.findById(id);
    if (!article) return res.status(404).json({ msg: "Article not found" });

    if (article.bannerImage) {
      await deleteFromCloudinary(article.bannerImage);
    }

    await ZuriMagazine.findByIdAndDelete(id);
    return res.status(200).json({ msg: "Article deleted" });
  } catch (error) {
    console.error("Error deleting article:", error);
    return res.status(500).json({ msg: "Internal Server Error" });
  }
};
//----------------------------------------------------------------------------------------------------------------------------
// Get all articles
export const getAllArticles = async (req, res) => {
  try {
    const articles = await ZuriMagazine.find().sort({ createdAt: -1 });
    return res.status(200).json({ data: articles, msg: "All articles fetched" });
  } catch (error) {
    console.error("Error fetching all articles:", error);
    return res.status(500).json({ msg: "Internal Server Error" });
  }
};

// Get single article by ID
export const getArticleById = async (req, res) => {
  try {
    const { id } = req.params;
    const article = await ZuriMagazine.findById(id);
    if (!article) return res.status(404).json({ msg: "Article not found" });

    return res.status(200).json({ data: article, msg: "Article fetched" });
  } catch (error) {
    console.error("Error fetching article by ID:", error);
    return res.status(500).json({ msg: "Internal Server Error" });
  }
};

// Get articles by category
export const getArticlesByCategory = async (req, res) => {
  try {
    const { category } = req.query;
    if (!category) return res.status(400).json({ msg: "Category is required" });

    const articles = await ZuriMagazine.find({ category: category.toLowerCase() }).sort({ createdAt: -1 });
    return res.status(200).json({ data: articles, msg: "Articles by category fetched" });
  } catch (error) {
    console.error("Error fetching articles by category:", error);
    return res.status(500).json({ msg: "Internal Server Error" });
  }
};

// Get all unique categories
export const getAllCategories = async (req, res) => {
  try {
    const categories = await ZuriMagazine.distinct("category");
    return res.status(200).json({ data: categories, msg: "Categories fetched" });
  } catch (error) {
    console.error("Error fetching categories:", error);
    return res.status(500).json({ msg: "Internal Server Error" });
  }
}


export const getAllBookmarkedArticles = async (req, res) => {
  try {
    const userId = req.user._id;

    const bookmarks = await Bookmark.find({ userId })
      .populate({
        path: 'articleId',
        model: 'ZuriMagazine',
        select: 'title authorName bannerImage category createdAt' // Select only the needed fields
      })
      .sort({ createdAt: -1 }); // Optional: latest bookmarks first

    // Extract populated article objects
    const articles = bookmarks
      .map(bookmark => bookmark.articleId)
      .filter(article => article !== null); // In case the article was deleted

    return res.status(200).json({
      msg: "Bookmarked articles fetched successfully",
      count: articles.length,
      data: articles
    });
  } catch (error) {
    console.error("Error fetching bookmarked articles:", error);
    return res.status(500).json({ msg: "Something went wrong while fetching bookmarks" });
  }
};

// switch between adding and removing from bookmarks
export const toggleBookmark = async (req, res) => {
  const { articleId } = req.params;
  const userId = req.user._id;

  try {
    const existing = await Bookmark.findOne({ userId, articleId });

    if (existing) {
      await Bookmark.deleteOne({ _id: existing._id });
      return res.status(200).json({ msg: "Bookmark removed" });
    }

    await Bookmark.create({ userId, articleId });
    return res.status(201).json({ msg: "Article bookmarked" });
  } catch (err) {
    console.error("Bookmark toggle error:", err);
    return res.status(500).json({ msg: "Something went wrong" });
  }
};