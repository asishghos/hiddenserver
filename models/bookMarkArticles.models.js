import mongoose from 'mongoose';

const bookmarkSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true
  },
  articleId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'ZuriMagazine',
    required: true,
    index: true
  }
}, {
  timestamps: true
});

// Ensure a user can bookmark an article only once
bookmarkSchema.index({ userId: 1, articleId: 1 }, { unique: true });

export const Bookmark = mongoose.model('Bookmark', bookmarkSchema);