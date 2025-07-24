import express from 'express';
import { toggleWishlistItem, getWishlistItems } from '../controllers/wishList.controllers.js';
import { verifyJWT } from '../middleware/auth.middleware.js';

const router = express.Router();

router.post('/add', verifyJWT, toggleWishlistItem);
router.get('/get', verifyJWT, getWishlistItems);

export default router;

// api/wishList/add
// api/wishlist/get