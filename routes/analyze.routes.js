import express from "express";
import {
  analyzeAuto,
  analyzeManual,
  analyzeHybrid,
} from "../controllers/analyze.controllers.js";
import { imageCheck } from "../controllers/checkfullbody.controllers.js";
import { upload } from "../middleware/multer.middleware.js";
import { verifyJWT } from "../middleware/auth.middleware.js"; // Your existing JWT middleware
import { User } from "../models/users.models.js"; // Import User model for additional routes

const router = express.Router();

// Apply authentication middleware to all routes
// router.use(verifyJWT);

// Auto analysis route - requires image upload and authentication
router.post("/auto", verifyJWT, upload.single("image"), analyzeAuto);

// Manual analysis route - requires only authentication (no image)
router.post("/manual", verifyJWT, analyzeManual);

// Hybrid analysis route - requires image upload and authentication
router.post("/hybrid", verifyJWT, upload.single("image"), analyzeHybrid);

router.post("/checkimage", upload.single("image"), imageCheck);
export default router;
