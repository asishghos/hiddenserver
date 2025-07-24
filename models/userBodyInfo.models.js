import mongoose from 'mongoose';

const colorSchema = new mongoose.Schema({
    name: String,
    hexCode: String
}, { _id: false });

const paletteSchema = new mongoose.Schema({
    colors: [colorSchema]
}, { _id: false });

const recommendedColorPalettesSchema = new mongoose.Schema({
    bright: paletteSchema,
    light: paletteSchema,
    neutral: paletteSchema
}, { _id: false });

const bodyShapeAnalysisSchema = new mongoose.Schema({
    bodyShape: {
        classification: String
    },
    gender: {
        classification: String
    },
    skinTone: {
        toneCategory: String,
        recommendedColorPalettes: recommendedColorPalettesSchema,
        reasoningProcess: String
    }
}, { _id: false });

const outfitRecommendationsSchema = new mongoose.Schema({
    items: [String]
}, { _id: false });

const outfitSuggestionsSchema = new mongoose.Schema({
    analysis: {
        body_type: String
    },
    recommendations: {
        tops: outfitRecommendationsSchema,
        bottoms: outfitRecommendationsSchema,
        dresses: outfitRecommendationsSchema,
        outerwear: outfitRecommendationsSchema,
        indian: outfitRecommendationsSchema,
        fabrics: outfitRecommendationsSchema
    },
    keywords: [[String]]
}, { _id: false });

const userBodyInfoSchema = new mongoose.Schema({
    userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true, unique: true },
    bodyShapeAnalysis: bodyShapeAnalysisSchema,
    outfitSuggestions: outfitSuggestionsSchema
}, {
    timestamps: true
});

export const UserBodyInfo = mongoose.model('UserBodyInfo', userBodyInfoSchema);
