import { User } from '../models/users.models.js';
import jwt from 'jsonwebtoken';

export const verifyJWT = async (req, res, next) => {
    try {
        const token = req.headers['authorization']?.split(' ')[1]; // Bearer <token>
        if (!token) {
            return res.status(401).json({ message: 'No token provided' });
        }
        const decodedToken = jwt.verify(token, process.env.JWT_ACCESS_SECRET);
        const user = await User.findById(decodedToken?._id).select('-refreshToken'); // Exclude sensitive fields
        if(!user) {
            return res.status(401).json({ message: 'Unauthorized' });
        }
        req.user = user; // Attach user to request object
        next();
    }
    catch (error) {
        console.error('JWT verification error:', error);
        return res.status(403).json({ message: 'something went wrong while verifing token' });
    }
};