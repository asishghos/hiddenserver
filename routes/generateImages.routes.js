import { Router } from "express";
import { generateImageForOccasion } from "../controllers/generateImages.controllers.js";
import { verifyJWT } from "../middleware/auth.middleware.js";

const router = Router();

router.post("/generateForOccasion", verifyJWT, generateImageForOccasion);

export default router;