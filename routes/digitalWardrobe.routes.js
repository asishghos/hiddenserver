import { Router } from 'express';
import { 
  addGarmentToDigitalWardrobe, 
  addGarmentToDigitalWardrobeByCategory,
  forceUploadMismatchedImages,
  updateGarment, 
  deleteGarment, 
  getGarmentsByCategory, 
//   getGarmentsByFabric, 
//   getGarmentsByOccasion, 
//   getGarmentsBySeason,
//   getGarmentsByColor,
  getCategoryCounts ,
  getGarmentDetails,
  filterGarments
} from "../controllers/digitalWardrobe.controllers.js";
import { verifyJWT } from "../middleware/auth.middleware.js";
import { upload } from "../middleware/multer.middleware.js";

const router = Router();

// Category counts
router.get("/categoryCounts", verifyJWT, getCategoryCounts);

// get garment detail by id
router.get("/getDetails/:garmentId", verifyJWT, getGarmentDetails);

// Add garments to wardrobe
router.post("/addTowardrobe", verifyJWT, upload.array("images", 5), addGarmentToDigitalWardrobe);
router.post("/addTowardrobeByCategory", verifyJWT, upload.array("images", 5), addGarmentToDigitalWardrobeByCategory);

// Force upload mismatched images
router.post("/forceUpload", verifyJWT, forceUploadMismatchedImages);

// Update and delete garments
router.put("/updateGarment/:garmentId", verifyJWT, updateGarment);
router.delete("/deleteGarment/:garmentId", verifyJWT, deleteGarment);

// Get garments by various filters (using query params instead of path params)
router.get("/garments/category", verifyJWT, getGarmentsByCategory);
router.get("/filterGarments", verifyJWT, filterGarments);

// router.get("/garments/fabric", verifyJWT, getGarmentsByFabric);
// router.get("/garments/occasion", verifyJWT, getGarmentsByOccasion);
// router.get("/garments/season", verifyJWT, getGarmentsBySeason);
// router.get("/garments/color", verifyJWT, getGarmentsByColor);

export default router;