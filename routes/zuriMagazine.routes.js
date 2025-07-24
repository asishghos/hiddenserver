import { Router } from "express";
import {
  getAllCategories,
  getArticlesByCategory,
  getAllArticles,
  addArticle,
  updateArticle,
  deleteArticle,
  getArticleById,
  getAllBookmarkedArticles,
  toggleBookmark,
} from "../controllers/zuriMagazine.controllers.js";
import { upload } from "../middleware/multer.middleware.js";
import { verifyJWT } from "../middleware/auth.middleware.js";

const router = Router();

// USER-FACING ROUTES
router.get("/allCategories", getAllCategories);
router.get("/articlesByCategory", getArticlesByCategory);
router.get("/article/:id", getArticleById);

// ADMIN ROUTES
router.post(
  "/addArticle",
  upload.fields([
    { name: "authorProfilePic", maxCount: 1 },
    { name: "bannerImage", maxCount: 1 },
  ]),
  addArticle
);
router.get("/allArticles", getAllArticles);
router.put("/updateArticle/:id", upload.single("bannerImage"), updateArticle);
router.delete("/deleteArticle/:id", deleteArticle);

router.post("/toggleBookmark/:articleId", verifyJWT, toggleBookmark);
router.get("/getBookmarks", verifyJWT, getAllBookmarkedArticles);

export default router;
