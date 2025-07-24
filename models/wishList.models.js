import mongoose from 'mongoose';

const wishlistSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        index: true,
        required: true
    },
    productId: { 
        type: String, 
        required: true 
    },
    productTitle: {
        type: String,
        required: true
    },
    productImage: {
        type: String
    },
    price: {
        type: String
    },
    originalPrice: {
        type: String
    },
    discountPercent: {
        type: String
    },
    platform: {
        type: String
    },
    rating: {
        type: String
    },
    productUrl: {
        type: String
    }
}, {timestamps: true});

wishlistSchema.index({ userId: 1, productId: 1 }, { unique: true });

export const Wishlist = mongoose.model('Wishlist', wishlistSchema);