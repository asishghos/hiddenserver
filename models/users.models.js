import mongoose from "mongoose";
import jwt from "jsonwebtoken";
import bcrypt from 'bcrypt';

const userSchema = new mongoose.Schema(
  {
    email: {
      type: String,
      required: true,
      unique: true,
      lowercase: true,
      trim: true,
      index: true,
      match: [
        /^\w+([.-]?\w+)*@\w+([.-]?\w+)*(\.\w{2,3})+$/,
        "Please enter a valid email",
      ],
    },
    fullName: {
      type: String,
      required: true,
      minlength: [2, "Username must be at least 3 characters"],
      maxlength: [20, "Username cannot exceed 20 characters"],
    },
    password: {
      type: String,
      required: [true, "Password is required!"],
      minlength: [3, "Password must be at least 6 characters"],
    },
    recoveryCode: {
      type: String
    },
    recoveryCodeExpiresAt: {
      type: Date
    },
    profilePicture: {
      type: String,
      default: "https://avatar.iran.liara.run/public/81"
    },
    userBodyInfo: {
      bodyShape: {
        type: String,
        enum: ['rectangle', 'hourglass', 'pear', 'apple', 'inverted triangle'],
      },
      undertone: {
        type: String,
        enum: ['cool', 'warm', 'neutral'],
      },
      height: {
        feet: {
          type: Number,
          min: 0,
        },
        inches: {
          type: Number,
          min: 0,
          max: 11,
        }
      }
    },
    refreshToken: {
      type: String,
    },
  },
  { timestamps: true }
);

userSchema.pre("save", async function (next) {          // pre is a mongoose hook
  if (!this.isModified("password")) return next();     // only if password field is modified then only pass to next

  this.password = await bcrypt.hash(this.password, 10)
  next()
})


userSchema.methods.isPasswordCorrect = async function (password) {        // custom method from mongoose
  return await bcrypt.compare(password, this.password)
}

userSchema.methods.generateAccessToken = function () {
  return jwt.sign(             // jwt is a bearer token(like a chabhi, whoever bears it, we return data to them)
    {                               //----\
      _id: this._id,                   //----\
      email: this.email,               //------>   payload (can't use arrow function since are using this)
      username: this.username,         //----/
      fullName: this.fullName          //----/
    },
    process.env.JWT_ACCESS_SECRET,
    {
      expiresIn: process.env.JWT_ACCESS_EXPIRES_IN
    }
  )
}

userSchema.methods.generateRefreshToken = function () {
  return jwt.sign(
    {
      _id: this._id,
    },
    process.env.JWT_REFRESH_SECRET,
    {
      expiresIn: process.env.JWT_REFRESH_EXPIRES_IN
    }

  )
}
// refresh token is saved on database as well as sent to user. when access token is expired we match tuhe refresh token from databse and user and keep him/her authenticated(they dont have to put their password all the time)


export const User = mongoose.model('User', userSchema)