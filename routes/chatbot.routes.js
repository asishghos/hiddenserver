import express from "express";
import { extractClothingKeywords } from "../controllers/chatbot.controllers.js";

const router = express.Router();

router.post("/extract", extractClothingKeywords);

export default router;
