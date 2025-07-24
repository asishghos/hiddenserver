import { User } from "../models/users.models.js";
import { DigitalWardrobe } from "../models/digitalWardrobe.models.js";
import jwt from "jsonwebtoken";
import {
  deleteFromCloudinary,
  uploadOnCloudinary,
} from "../utils/cloudinary.js";
import crypto from "crypto";
import { sendEmail } from "../utils/sendEmail.js";

// const generateAccessAndRefreshTokens = async (userId) => {
//   try {
//     //finding user from database by _id
//     const user = await User.findById(userId);

//     //generating Token
//     const refreshToken = user.generateRefreshToken();
//     const accessToken = user.generateAccessToken();

//     //saving refreshToken into the database(access token is not added to database)
//     user.refreshToken = refreshToken;
//     await user.save({ validateBeforeSave: false }); //automatically mongoose model(password) kick in so we pass validateBeforeSave to avoid this

//     return { accessToken, refreshToken };
//   } catch (error) {
//     console.log("Error generating tokens:", error);
//     return {
//       success: false,
//       error: error.message,
//     };
//   }
// };
// register user
export const registerUser = async (req, res) => {
  try {
    const { fullName, email, password } = req.body;

    // Validation
    if ([fullName, email, password].some((field) => field?.trim() === "")) {
      return res.status(400).json({ msg: "All fields are required" });
    }

    // Check for existing user
    const existedUser = await User.findOne({ $or: [{ email }] });
    if (existedUser) {
      return res.status(409).json({ msg: "User already exists" });
    }

    // Handle optional profile picture
    let profilePictureUrl = "";
    const profilePicFile = req.files?.profilePicture?.[0];

    if (profilePicFile) {
      const cloudinaryResult = await uploadOnCloudinary(profilePicFile.path); // it returns the URL itself
      profilePictureUrl = cloudinaryResult || "";
    }

    // Create new user
    const user = await User.create({
      fullName,
      email,
      password,
      profilePicture: profilePictureUrl,
    });

    if (!user) {
      return res
        .status(500)
        .json({ msg: "Something went wrong while registering the user" });
    }

    return res
      .status(201)
      .json({ data: user, msg: "User registered successfully" });
  } catch (error) {
    console.error("Registration error:", error);
    return res.status(500).json({ msg: "Internal server error" });
  }
};

export const loginUser = async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ msg: "All fields are required" });
    }

    const user = await User.findOne({ email });

    if (!user) {
      return res.status(404).json({ msg: "User not found" });
    }

    const isPasswordCorrect = await user.isPasswordCorrect(password);
    if (!isPasswordCorrect) {
      return res.status(401).json({ msg: "Invalid credentials" });
    }

    const { accessToken, refreshToken } = await generateAccessAndRefreshTokens(
      user._id
    );

    const loggedInUser = await User.findById(user._id).select(
      "-password -refreshToken"
    );

    const options = {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: process.env.NODE_ENV === "production" ? "none" : "lax",
      maxAge: 24 * 60 * 60 * 1000, // 24 hours
    };

    const response = {
      data: loggedInUser,
      msg: "User Logged In Successfully",
      tokens: {
        accessToken,
        refreshToken,
      },
      success: true,
    };

    return res
      .status(200)
      .cookie("accessToken", accessToken, options)
      .cookie("refreshToken", refreshToken, options)
      .json(response);
  } catch (error) {
    console.error("Login error:", error);
    return res.status(500).json({
      msg: "Internal server error",
      error: error.message,
      success: false,
    });
  }
};

const generateAccessAndRefreshTokens = async (userId) => {
  try {
    if (!process.env.JWT_ACCESS_SECRET || !process.env.JWT_REFRESH_SECRET) {
      throw new Error("JWT secrets not configured");
    }

    const user = await User.findById(userId);

    if (!user) {
      throw new Error("User not found");
    }
    const accessToken = user.generateAccessToken();
    const refreshToken = user.generateRefreshToken();

    if (!accessToken || !refreshToken) {
      throw new Error("Token generation failed");
    }
    user.refreshToken = refreshToken;
    await user.save({ validateBeforeSave: false });

    return { accessToken, refreshToken };
  } catch (error) {
    console.error("Error generating tokens:", error);
    throw error;
  }
};

export const logoutUser = async (req, res) => {
  await User.findByIdAndUpdate(
    req.user._id,
    {
      $unset: {
        refreshToken: 1,
      },
    },
    {
      new: true, // mongoDB response will be new one updated
    }
  );

  const options = {
    httpOnly: true,
    secure: true,
  };

  return res
    .status(200)
    .cookie("accessToken", null, options)
    .cookie("refreshToken", null, options)
    .json({ msg: "User Logged Out Successfully" });
};

// refresh access token
export const refreshAccessToken = async (req, res) => {
  try {
    const incomingRefreshToken = req.body.refreshToken;
    if (!incomingRefreshToken) {
      return res.status(400).json({ msg: "Refresh token is required" });
    }
    const decodedToken = jwt.verify(
      incomingRefreshToken,
      process.env.JWT_REFRESH_SECRET
    );
    const user = await user.findById(decodedToken._id);

    if (!user) {
      return res.status(404).json({ msg: "User not found" });
    }
    if (user.refreshToken !== incomingRefreshToken) {
      return res.status(403).json({ msg: "Invalid refresh token" });
    }
    const options = {
      httpOnly: true,
      secure: true,
    };

    const { accessToken, newRefreshToken } =
      await generateAccessAndRefreshTokens(user._id);

    return res
      .status(200)
      .cookie("accessToken", accessToken, options)
      .cookie("refreshToken", newRefreshToken, options)
      .json({
        msg: "Access token refreshed successfully",
        accessToken,
        newRefreshToken,
      });
  } catch (error) {
    return res
      .status(500)
      .json({ msg: "Something went wrong while refreshing access token" });
  }
};

// change user password
export const changeUserPassword = async (req, res) => {
  try {
    const user = req.user;
    const { oldPassword, newPassword } = req.body;

    if (!oldPassword || !newPassword) {
      return res.status(400).json({ msg: "All fields are required" });
    }

    const isPasswordCorrect = await user.isPasswordCorrect(oldPassword);
    if (!isPasswordCorrect) {
      return res.status(401).json({ msg: "Invalid credentials" });
    }

    user.password = newPassword;
    await user.save({ validateBeforeSave: true });
    return res.status(200).json({ msg: "Password changed successfully" });
  } catch (error) {
    return res
      .status(500)
      .json({ msg: "Something went wrong while changing password" });
  }
};

// get user profile
export const getUserProfile = async (req, res) => {
  try {
    const user = req.user;

    if (!user) {
      return res
        .status(404)
        .json({ msg: "User not found while fetching profile" });
    }

    const userId = user._id;

    const wardrobe = await DigitalWardrobe.findOne({ userId });

    let totalGarments = 0;
    let latestGarmentsImages = [];

    if (wardrobe && wardrobe.uploadedImages?.length) {
      // Count garments
      totalGarments = wardrobe.uploadedImages.reduce((total, image) => {
        return total + (image.garments?.length || 0);
      }, 0);

      // Get latest 3 images (reversed for most recent)
      latestGarmentsImages = wardrobe.uploadedImages
        .slice(-3)
        .reverse()
        .map((image) => ({ imageUrl: image.imageUrl }));
    }

    return res.status(200).json({
      data: user,
      closetStats: totalGarments,
      newAdditions: latestGarmentsImages,
      msg: "User profile fetched successfully",
    });
  } catch (error) {
    console.error("Error fetching profile:", error);
    return res
      .status(500)
      .json({ msg: "Something went wrong while fetching user profile" });
  }
};

// get userName
export const getUserFullName = async (req, res) => {
  try {
    const user = req.user;
    const userName = user.fullName;
    return res.status(200).json({ data: userName });
  } catch (error) {
    return res
      .status(500)
      .json({ msg: "Something went wrong while fetching user name" });
  }
};

// update user profile
export const updateUserProfile = async (req, res) => {
  try {
    const user = req.user;
    if (!user) {
      return res
        .status(404)
        .json({ msg: "User not found while updating profile" });
    }

    const { fullName, email, bodyShape, undertone, feet, inches, location } =
      req.body;

    const profilePicFile = req.files?.profilePicture?.[0];

    if (
      !fullName &&
      !email &&
      !bodyShape &&
      !undertone &&
      !feet &&
      !inches &&
      !location &&
      !profilePicFile
    ) {
      return res
        .status(400)
        .json({ msg: "At least one field or profile picture is required" });
    }

    // Update basic fields
    if (fullName) user.fullName = fullName;
    if (email) user.email = email;
    if (bodyShape) user.userBodyInfo.bodyShape = bodyShape;
    if (undertone) user.userBodyInfo.undertone = undertone;
    if (feet !== undefined) user.userBodyInfo.height.feet = feet;
    if (inches !== undefined) user.userBodyInfo.height.inches = inches;
    // if (location) user.location = location;

    // Upload new profile picture if provided
    if (profilePicFile) {
      const uploadedUrl = await uploadOnCloudinary(profilePicFile.path);
      if (!uploadedUrl) {
        return res
          .status(500)
          .json({ msg: "Failed to upload profile picture" });
      }

      // Delete old picture
      if (user.profilePicture) {
        await deleteFromCloudinary(user.profilePicture);
      }

      user.profilePicture = uploadedUrl;
    }

    await user.save({ validateBeforeSave: true });

    return res
      .status(200)
      .json({ msg: "User profile updated successfully", data: user });
  } catch (error) {
    console.error("Error updating profile:", error);
    return res
      .status(500)
      .json({ msg: "Something went wrong while updating user profile" });
  }
};

// forgot password
export const forgotPassword = async (req, res) => {
  try {
    const { email } = req.body;

    if (!email) return res.status(400).json({ msg: "Email is required" });

    const user = await User.findOne({ email });
    if (!user) return res.status(404).json({ msg: "User not found" });

    const recoveryCode = crypto.randomInt(100000, 999999).toString(); // 6-digit code
    const expires = new Date(Date.now() + 10 * 60 * 1000); // expires in 10 minutes

    user.recoveryCode = recoveryCode;
    user.recoveryCodeExpiresAt = expires;
    await user.save();

    // Send email (build your utility)
    await sendEmail({
      to: user.email,
      subject: "Your password reset code",
      text: `Use this code to reset your password: ${recoveryCode}. It will expire in 10 minutes.`,
    });

    return res.status(200).json({ msg: "Recovery code sent to email" });
  } catch (err) {
    console.error("Forgot password error:", err);
    return res.status(500).json({ msg: "Internal Server Error" });
  }
};

// verify recovery-code
export const verifyRecoveryCode = async (req, res) => {
  try {
    const { email, code } = req.body;
    const user = await User.findOne({ email });

    if (!user || !user.recoveryCode || !user.recoveryCodeExpiresAt) {
      return res.status(400).json({ msg: "Invalid or expired recovery code" });
    }

    if (user.recoveryCode !== code) {
      return res.status(400).json({ msg: "Incorrect recovery code" });
    }

    if (user.recoveryCodeExpiresAt < new Date()) {
      return res.status(400).json({ msg: "Recovery code has expired" });
    }

    return res.status(200).json({ msg: "Recovery code verified" });
  } catch (err) {
    console.error("Recovery code verification error:", err);
    return res.status(500).json({ msg: "Internal Server Error" });
  }
};

// reset password
export const resetPassword = async (req, res) => {
  try {
    const { email, newPassword, confirmPassword } = req.body;

    if (newPassword !== confirmPassword) {
      return res.status(400).json({ msg: "Passwords do not match" });
    }

    const user = await User.findOne({ email });

    if (!user || user.recoveryCodeExpiresAt < new Date()) {
      return res.status(400).json({ msg: "Invalid or expired recovery code" });
    }

    user.password = newPassword; // Will be hashed in pre-save middleware
    user.recoveryCode = undefined;
    user.recoveryCodeExpiresAt = undefined;

    await user.save();

    return res.status(200).json({ msg: "Password reset successfully" });
  } catch (err) {
    console.error("Reset password error:", err);
    return res.status(500).json({ msg: "Internal Server Error" });
  }
};
