import { Router } from "express";
import { addStyledImageToEvent } from "../controllers/addImageToEvent.controllers.js";
import { verifyJWT } from "../middleware/auth.middleware.js";

const router = Router();

// for single day eve
router.post("/addImageToEvent/:eventId", verifyJWT, addStyledImageToEvent);
// for multi day event
router.post("/addImageToEvent/:eventId/:dayEventId", verifyJWT, addStyledImageToEvent);

export default router;
