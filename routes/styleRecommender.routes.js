import {Router} from 'express';
import {styleRecommenderController} from "../controllers/styleRecommender.controllers.js";
import {upload} from "../middleware/multer.middleware.js";
import {verifyJWT} from "../middleware/auth.middleware.js";

const router = Router();

router.post('/', verifyJWT, upload.array('images', 4), styleRecommenderController);

export default router;