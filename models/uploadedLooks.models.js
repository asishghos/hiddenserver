import mongoose from "mongoose";

const uploadedLooksSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "User",
        required: true
    },
    imageUrl: {
        type: String,
        required: true,
        trim: true
    },
    title: {
        type: String,
        required: true,
        trim: true
    }
}, {
    timestamps: true
});

uploadedLooksSchema.index({ userId: 1, imageUrl: 1 }, { unique: true });

export const UploadedLooks = mongoose.model("UploadedLooks", uploadedLooksSchema);