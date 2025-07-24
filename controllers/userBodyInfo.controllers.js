import { UserBodyInfo } from "../models/userBodyInfo.models.js";

export const createUserBodyInfo = async (req, res) => {
    const userId = req.user._id;
    const bodyInfoData = req.body;

    try {
        // Check if entry already exists
        const existing = await UserBodyInfo.findOne({ userId });
        if (existing) {
            return res.status(400).json({ message: 'Body info already exists. Use update instead.' });
        }

        const newBodyInfo = await UserBodyInfo.create({ userId, ...bodyInfoData });
        return res.status(201).json({ message: 'User body info created successfully', data: newBodyInfo });
    } catch (error) {
        return res.status(500).json({ message: 'Error creating body info', error: error.message });
    }
};

export const updateUserBodyInfo = async (req, res) => {
    const userId = req.user._id;
    const bodyInfoData = req.body;

    try {
        const updatedBodyInfo = await UserBodyInfo.findOneAndUpdate(
            { userId },
            bodyInfoData,
            { new: true }
        );

        if (!updatedBodyInfo) {
            return res.status(404).json({ message: 'No existing body info to update' });
        }

        return res.status(200).json({ message: 'User body info updated successfully', data: updatedBodyInfo });
    } catch (error) {
        return res.status(500).json({ message: 'Error updating body info', error: error.message });
    }
};

export const getUserBodyInfo = async (req, res) => {
    const userId = req.user._id;

    try {
        const bodyInfo = await UserBodyInfo.findOne({ userId });
        if (!bodyInfo) {
            return res.status(404).json({ message: 'No body info found for user' });
        }
        return res.status(200).json({ data: bodyInfo });
    } catch (error) {
        return res.status(500).json({ message: 'Error fetching body info', error: error.message });
    }
};

export const deleteUserBodyInfo = async (req, res) => {
    const userId = req.user._id;

    try {
        const deleted = await UserBodyInfo.findOneAndDelete({ userId });
        if (!deleted) {
            return res.status(404).json({ message: 'No body info found to delete' });
        }
        return res.status(200).json({ message: 'User body info deleted successfully' });
    } catch (error) {
        return res.status(500).json({ message: 'Error deleting body info', error: error.message });
    }
};