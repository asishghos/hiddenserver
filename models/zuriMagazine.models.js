import mongoose from 'mongoose';

const zuriMagazineSchema = new mongoose.Schema({
    authorProfilePic: {
        type: String,
        default: "https://avatar.iran.liara.run/public/97"
    },
    authorName: {
        type: String,
        required: true
    },
    category: {
        type: String,
        required: true
    },
    title: {
        type: String,
        required: true
    },
    content: {
        type: String,
        required: true
    },
    subTitle: {
        type: String
    },
    bannerImage: {
        type: String
    },
    tags: {
        type: [String],
        default: []
    }
}, { timestamps: true });

zuriMagazineSchema.pre('save', function (next) {
    if (this.category) {
        this.category = this.category.toLowerCase();
    }
    next();
});

zuriMagazineSchema.pre('findOneAndUpdate', function (next) {
    const update = this.getUpdate();
    if (update?.category) {
        update.category = update.category.toLowerCase();
        this.setUpdate(update);
    }
    next();
});


export const ZuriMagazine = mongoose.model('ZuriMagazine', zuriMagazineSchema);