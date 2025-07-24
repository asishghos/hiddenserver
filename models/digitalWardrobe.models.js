import mongoose from "mongoose";

const GarmentPieceSchema = new mongoose.Schema(
  {
    itemName: {
      type: String,
      required: true,
      trim: true,
      maxlength: 100,
    },
    category: {
      type: String,
      enum: [
        "Tops",
        "Bottoms",
        "Ethnic",
        "Dresses",
        "co-ord set",
        "Swimwear",
        "Footwear",
        "Accessories",
      ],
      required: true,
    },
    color: {
      name: {
        type: String,
        required: true,
        trim: true,
        maxlength: 50,
      },
      hex: {
        type: String,
        required: true,
        trim: true,
        validate: {
          validator: function (v) {
            return /^#([A-Fa-f0-9]{6})$/.test(v);
          },
          message: "Invalid hex color format. Example: #FF5733",
        },
      },
    },

    fabric: {
      type: String,
      enum: [
        "Cotton",
        "Linen",
        "Silk",
        "Wool",
        "Denim",
        "Polyester",
        "Rayon",
        "Velvet",
        "Chiffon",
        "Georgette",
        "Net",
        "Satin",
        "Tulle",
        "Mixed",
        "Knit",
        "Other",
      ],
      required: true,
    },
    occasion: [
      {
        type: String,
        trim: true,
        maxlength: 30,
      },
    ],
    season: [
      {
        type: String,
        enum: ["Summer", "Winter", "Monsoon", "Autumn", "Spring", "All Season"],
        required: true,
      },
    ],
  },
  {
    _id: true, // Each garment piece gets its own ID
    timestamps: false, // We'll use the parent image's timestamp
  }
);

// Add validation for occasion and season limits
GarmentPieceSchema.pre("validate", function () {
  if (this.occasion && this.occasion.length > 3) {
    this.occasion = this.occasion.slice(0, 3);
  }
  if (this.season && this.season.length > 2) {
    this.season = this.season.slice(0, 2);
  }
});

const UploadedImageSchema = new mongoose.Schema(
  {
    imageUrl: {
      type: String,
      required: true,
      validate: {
        validator: function (v) {
          return /^https?:\/\/.+/.test(v);
        },
        message: "Image URL must be a valid HTTP/HTTPS URL",
      },
    },
    imageHash: {
      type: String,
      required: true,
      unique: false, // Not globally unique, but unique within a user's wardrobe
      length: 64, // SHA-256 hash length
    },
    garments: {
      type: [GarmentPieceSchema],
      required: true,
      validate: {
        validator: function (v) {
          return v && v.length > 0;
        },
        message: "At least one garment must be present",
      },
    },
    createdAt: {
      type: Date,
      default: Date.now,
      index: true, // For efficient querying by date
    },
  },
  {
    _id: true, // Each uploaded image gets its own ID
  }
);

// Compound index to ensure unique images per user
UploadedImageSchema.index({ imageHash: 1 }, { unique: false });

const DigitalWardrobeSchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      unique: true,
      required: true,
      index: true, // For efficient user lookups
    },
    uploadedImages: {
      type: [UploadedImageSchema],
      default: [],
    },
    totalGarments: {
      type: Number,
      default: 0,
    },
  },
  {
    timestamps: true, // This will automatically manage createdAt and updatedAt
  }
);

// Compound index to ensure unique image hashes per user
DigitalWardrobeSchema.index({ userId: 1, "uploadedImages.imageHash": 1 });

// Pre-save middleware to update totalGarments count
DigitalWardrobeSchema.pre("save", function () {
  if (this.isModified("uploadedImages")) {
    this.totalGarments = this.uploadedImages.reduce((total, image) => {
      return total + (image.garments ? image.garments.length : 0);
    }, 0);
  }
});

export const DigitalWardrobe = mongoose.model(
  "DigitalWardrobe",
  DigitalWardrobeSchema
);
