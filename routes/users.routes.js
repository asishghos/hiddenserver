import { verifyJWT } from "../middleware/auth.middleware.js";
import { Router } from "express";
import {
  registerUser,
  loginUser,
  logoutUser,
  getUserProfile,
  changeUserPassword,
  updateUserProfile,
  refreshAccessToken,
  getUserFullName,
  forgotPassword,
  verifyRecoveryCode,
  resetPassword,
} from "../controllers/users.controllers.js";
import { upload } from "../middleware/multer.middleware.js";

const router = Router();

// Register a new user
router.post(
  "/register",
  upload.fields([
    {
      name: "profilePicture",
      maxCount: 1,
    },
  ]),
  registerUser
);

// router.post("/register", registerUser);
// Login a user
router.post("/login", loginUser);
// Logout a user
router.post("/logout", verifyJWT, logoutUser);
// Get user profile
router.get("/profile", verifyJWT, getUserProfile);
// Get user fullName
router.get("/userName", verifyJWT, getUserFullName);
// Change user password
router.patch("/changePassword", verifyJWT, changeUserPassword);
// Update user profile
router.patch(
  "/updateProfile",
  verifyJWT,
  upload.fields([
    {
      name: "profilePicture",
      maxCount: 1,
    },
  ]),
  updateUserProfile
);
// Refresh access token
router.post("/refreshToken", refreshAccessToken);
// forgot password
router.post("/forgotPassword", forgotPassword);
// verify recovery code
router.post("/verifyRecoveryCode", verifyRecoveryCode);
// reset password
router.post("/resetPassword", resetPassword);

export default router;
