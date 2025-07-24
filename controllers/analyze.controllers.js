import { PythonShell } from "python-shell";
import axios from "axios";
import fs from "fs";
import { User } from "../models/users.models.js"; // Adjust path as needed

const prototypePrompt1 = fs.readFileSync("prompts/prompt1.txt", "utf8");

// Use environment variable for API key
const OPENAI_API_KEY =
  process.env.OPENAI_API_KEY;

async function runPoseDetector(imagePath) {
  return new Promise((resolve, reject) => {
    // Validate file exists
    if (!fs.existsSync(imagePath)) {
      return reject(new Error("Image file not found"));
    }

    const imageBuffer = fs.readFileSync(imagePath);
    const base64Image = imageBuffer.toString("base64");

    const pyshell = new PythonShell("tools/pose_detector.py");
    let output = "";

    pyshell.on("message", (message) => {
      output += message + "\n";
    });

    pyshell.on("stderr", (stderr) => {
      console.error("Python stderr:", stderr);
    });

    pyshell.send(JSON.stringify({ image: base64Image }));

    pyshell.end((err) => {
      if (err) return reject(err);
      resolve(output.trim());
    });
  });
}

async function runToneDetector(imagePath, landmarkResponse) {
  return new Promise((resolve, reject) => {
    // Validate file exists
    if (!fs.existsSync(imagePath)) {
      return reject(new Error("Image file not found"));
    }

    const imageBuffer = fs.readFileSync(imagePath);
    const base64Image = imageBuffer.toString("base64");

    const pyshell = new PythonShell("tools/skintone_detector.py");
    let output = "";

    pyshell.on("message", (message) => {
      output += message + "\n";
    });

    pyshell.on("stderr", (stderr) => {
      console.error("Python stderr:", stderr);
    });

    pyshell.send(
      JSON.stringify({
        image: base64Image,
        keypoints_text: landmarkResponse,
      })
    );

    pyshell.end((err) => {
      if (err) return reject(err);
      resolve(output.trim());
    });
  });
}

async function getBodyShapeFromGPT(landmarkResponse, toneResponse, imagePath) {
  const messages = [
    {
      role: "system",
      content: prototypePrompt1,
    },
    {
      role: "user",
      content: [
        {
          type: "text",
          text: `Human Analysis Request
          ## Input Data Analysis
          I need a comprehensive analysis of the human subject using the following data:
          ### 1. Pose Landmarks Data/ Body shape: ${JSON.stringify(landmarkResponse)}
          ### 2. Skin Tone Detection Results: ${JSON.stringify(toneResponse)}
          ${
            imagePath
              ? "### 3. Reference Image is included."
              : "### 3. No image was provided."
          }`,
        },
      ],
    },
  ];

  if (imagePath && fs.existsSync(imagePath)) {
    const imageBuffer = fs.readFileSync(imagePath);
    const base64Image = imageBuffer.toString("base64");
    messages[1].content.push({
      type: "image_url",
      image_url: {
        url: `data:image/jpeg;base64,${base64Image}`,
      },
    });
  }

  try {
    const response = await axios.post(
      "https://api.openai.com/v1/chat/completions",
      {
        model: "gpt-4-turbo",
        messages: messages,
        temperature: 0.3,
        response_format: { type: "json_object" },
      },
      {
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        },
      }
    );

    return JSON.parse(response.data.choices[0].message.content);
  } catch (error) {
    console.error("OpenAI API error:", error.response?.data || error.message);
    throw new Error("Failed to analyze body shape");
  }
}

// Helper function to extract and normalize body shape from GPT response
function extractBodyShape(gptResponse) {
  const validShapes = ['rectangle', 'hourglass', 'pear', 'apple', 'inverted triangle'];
  
  // Look for body shape in various possible fields
  let bodyShape = gptResponse.bodyShape || 
                  gptResponse.body_shape || 
                  gptResponse.shape ||
                  gptResponse.bodyType ||
                  gptResponse.body_type;

  if (typeof bodyShape === 'string') {
    bodyShape = bodyShape.toLowerCase().trim();
    
    // Handle common variations
    if (bodyShape.includes('inverted') || bodyShape.includes('triangle')) {
      bodyShape = 'inverted triangle';
    }
    
    // Check if it's a valid shape
    if (validShapes.includes(bodyShape)) {
      return bodyShape;
    }
  }
  
  return null;
}

// Helper function to extract and normalize skin undertone from GPT response
function extractUndertone(gptResponse) {
  const validUndertones = ['cool', 'warm', 'neutral'];
  
  // Look for undertone in various possible fields
  let undertone = gptResponse.undertone || 
                  gptResponse.skinUndertone || 
                  gptResponse.skin_undertone ||
                  gptResponse.skinTone ||
                  gptResponse.skin_tone ||
                  gptResponse.tone;

  if (typeof undertone === 'string') {
    undertone = undertone.toLowerCase().trim();
    
    // Check if it's a valid undertone
    if (validUndertones.includes(undertone)) {
      return undertone;
    }
  }
  
  return null;
}

// Helper function to save user body info
async function saveUserBodyInfo(userId, bodyShape, undertone) {
  try {
    const updateData = {};
    
    if (bodyShape) {
      updateData['userBodyInfo.bodyShape'] = bodyShape;
    }
    
    if (undertone) {
      updateData['userBodyInfo.undertone'] = undertone;
    }
    
    if (Object.keys(updateData).length > 0) {
      const updatedUser = await User.findByIdAndUpdate(
        userId,
        { $set: updateData },
        { new: true, runValidators: true }
      );
      
      if (!updatedUser) {
        throw new Error('User not found');
      }
      
      console.log('User body info updated successfully:', {
        userId,
        bodyShape,
        undertone
      });
      
      return updatedUser;
    }
    
    return null;
  } catch (error) {
    console.error('Error saving user body info:', error);
    throw error;
  }
}

export const analyzeAuto = async (req, res) => {
  const imagePath = req.file?.path;
  const userId = req.user?._id; // Assuming you have user authentication middleware
  
  if (!imagePath) return res.status(400).json({ error: "No image uploaded" });
  if (!userId) return res.status(401).json({ error: "User authentication required" });

  try {
    const landmarkResponse = await runPoseDetector(imagePath);
    const toneResponse = await runToneDetector(imagePath, landmarkResponse);

    let bodyShapeResult = await getBodyShapeFromGPT(
      landmarkResponse,
      toneResponse,
      imagePath
    );

    console.log("Tone response:", toneResponse);
    console.log("Keypoints response:", landmarkResponse);
    console.log("GPT response 1: ", bodyShapeResult);

    // Extract and save body shape and undertone
    const bodyShape = extractBodyShape(bodyShapeResult);
    const undertone = extractUndertone(bodyShapeResult);
    
    let updatedUser = null;
    if (bodyShape || undertone) {
      updatedUser = await saveUserBodyInfo(userId, bodyShape, undertone);
    }

    res.json({
      bodyShapeResult,
      saved: {
        bodyShape: bodyShape || 'Not detected',
        undertone: undertone || 'Not detected',
        updated: !!updatedUser
      }
    });
  } catch (error) {
    console.error("analyzeAuto error:", error);
    res.status(500).json({ error: error.message || "Processing failed" });
  } finally {
    // Safe file cleanup
    if (imagePath && fs.existsSync(imagePath)) {
      try {
        fs.unlinkSync(imagePath);
      } catch (cleanupError) {
        console.error("File cleanup error:", cleanupError);
      }
    }
  }
};

export const analyzeManual = async (req, res) => {
  const userId = req.user?._id; // Assuming you have user authentication middleware
  
  if (!userId) return res.status(401).json({ error: "User authentication required" });

  try {
    // Input validation
    if (!req.body.body_shape || !req.body.skin_tone) {
      return res
        .status(400)
        .json({ error: "Missing required fields: body_shape and skin_tone" });
    }

    const requestData = {
      body_shape: req.body.body_shape,
      gender: req.body.gender || "female", // Allow gender to be specified
      skin_tone: req.body.skin_tone,
    };

    let bodyShapeResult = await getBodyShapeFromGPT(
      JSON.stringify(requestData),
      JSON.stringify(requestData),
      null
    );

    // For manual input, we can directly use the provided values
    const bodyShape = req.body.body_shape.toLowerCase().trim();
    const undertone = req.body.skin_tone.toLowerCase().trim();
    
    // Validate against schema enums
    const validShapes = ['rectangle', 'hourglass', 'pear', 'apple', 'inverted triangle'];
    const validUndertones = ['cool', 'warm', 'neutral'];
    
    const validBodyShape = validShapes.includes(bodyShape) ? bodyShape : null;
    const validUndertone = validUndertones.includes(undertone) ? undertone : null;
    
    let updatedUser = null;
    if (validBodyShape || validUndertone) {
      updatedUser = await saveUserBodyInfo(userId, validBodyShape, validUndertone);
    }

    res.json({
      bodyShapeResult,
      saved: {
        bodyShape: validBodyShape || 'Invalid body shape',
        undertone: validUndertone || 'Invalid undertone',
        updated: !!updatedUser
      }
    });
  } catch (error) {
    console.error("analyzeManual error:", error);
    res.status(500).json({ error: error.message || "Processing failed" });
  }
};

export const analyzeHybrid = async (req, res) => {
  const imagePath = req.file?.path;
  const userId = req.user?._id; // Assuming you have user authentication middleware
  
  if (!imagePath) return res.status(400).json({ error: "No image uploaded" });
  if (!userId) return res.status(401).json({ error: "User authentication required" });
  if (!req.body.body_shape) {
    return res
      .status(400)
      .json({ error: "Missing required fields: body_shape" });
  }

  try {
    const landmarkResponse = await runPoseDetector(imagePath);
    const toneResponse = await runToneDetector(imagePath, landmarkResponse);

    const requestData = {
      body_shape: req.body.body_shape,
      gender: req.body.gender || "female",
    };

    let bodyShapeResult = await getBodyShapeFromGPT(
      requestData,
      toneResponse,
      imagePath
    );

    console.log("Tone response:", toneResponse);
    console.log("Keypoints response:", landmarkResponse);
    console.log("GPT response 1: ", bodyShapeResult);

    // For hybrid, use manual body shape and detected undertone
    const bodyShape = req.body.body_shape.toLowerCase().trim();
    const undertone = extractUndertone(bodyShapeResult);
    
    // Validate body shape
    const validShapes = ['rectangle', 'hourglass', 'pear', 'apple', 'inverted triangle'];
    const validBodyShape = validShapes.includes(bodyShape) ? bodyShape : null;
    
    let updatedUser = null;
    if (validBodyShape || undertone) {
      updatedUser = await saveUserBodyInfo(userId, validBodyShape, undertone);
    }

    res.json({
      bodyShapeResult,
      saved: {
        bodyShape: validBodyShape || 'Invalid body shape',
        undertone: undertone || 'Not detected',
        updated: !!updatedUser
      }
    });
  } catch (error) {
    console.error("analyzeHybrid error:", error);
    res.status(500).json({ error: error.message || "Processing failed" });
  } finally {
    // Safe file cleanup
    if (imagePath && fs.existsSync(imagePath)) {
      try {
        fs.unlinkSync(imagePath);
      } catch (cleanupError) {
        console.error("File cleanup error:", cleanupError);
      }
    }
  }
};