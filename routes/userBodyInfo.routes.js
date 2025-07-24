import { createUserBodyInfo, updateUserBodyInfo, deleteUserBodyInfo, getUserBodyInfo } from "../controllers/userBodyInfo.controllers.js";
import { verifyJWT } from "../middleware/auth.middleware.js";
import { Router } from "express";

const router = Router();

router.post("/createUserInfo", verifyJWT, createUserBodyInfo); // Create or update user body info
router.put("/updateUserInfo", verifyJWT, updateUserBodyInfo); // Update user body info
router.get("/getUserInfo", verifyJWT, getUserBodyInfo); // Get user body info
router.delete("/deleteUserInfo", verifyJWT, deleteUserBodyInfo); // Delete user body info

export default router;