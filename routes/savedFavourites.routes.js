import { Router } from "express";
import { verifyJWT } from "../middleware/auth.middleware.js";
import {
  addToSavedFavourites,
  getSavedFavourites,
  deleteSavedFavourite,
} from "../controllers/savedFavourites.controllers.js";
import { upload } from "../middleware/multer.middleware.js";

const router = Router();

router.post(
  "/addFavourite",
  verifyJWT,
  upload.array("files", 5),
  addToSavedFavourites
);
router.get("/getFavourites", verifyJWT, getSavedFavourites);
router.delete("/deleteFavourite/:favouriteId", verifyJWT, deleteSavedFavourite);

export default router;
