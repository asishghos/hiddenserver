import { PythonShell } from "python-shell";
import axios from "axios";
import fs from "fs";
import { User } from "../models/users.models.js"; // Adjust path as needed

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
      content: `You are an advanced expert system designed to analyze human body shape, gender, and skin tone using MediaPipe pose Landmark, and skin tone detection results. Your analysis will provide comprehensive insights based on scientific ratios and established classification systems.

## 1. Body Shape Analysis
### Required Keypoint Data
Extract these MediaPipe pose Landmarks from the input data:
- Left shoulder (x,y)
- Right shoulder (x,y)
- Left hip (x,y)
- Right hip (x,y)

### Handling Missing Data
If body landmark data is not provided or is incomplete:
- Use the directly provided \`body_shape\` value from the input JSON
- NEVER return "Undefined" as a body shape classification
- If no shape is specified, default to "Hourglass" with a lower confidence score (60%)

### Measurement Calculations
When landmark data is available:
\`\`\`
shoulder_width = |right_shoulder.x - left_shoulder.x|
hip_width = |right_hip.x - left_hip.x|
vertical_torso_length = average(|right_shoulder.y - right_hip.y|, |left_shoulder.y - left_hip.y|)

hip_midpoint = ((left_hip.x + right_hip.x)/2, (left_hip.y + right_hip.y)/2)
shoulder_midpoint = ((left_shoulder.x + right_shoulder.x)/2, (left_shoulder.y + right_shoulder.y)/2)

waist_x = (right_hip.x + left_hip.x)/2
waist_y = ((left_hip.y + right_hip.y)/2 + (left_shoulder.y + right_shoulder.y)/2 ) * 0.5 
waist_point = (waist_x, waist_y)
inferred_waist_y = shoulder_midpoint.y + 0.5 × vertical_torso_length
inferred_waist_x = average(left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x)
inferred_waist_point = (inferred_waist_x, inferred_waist_y)
\`\`\`

### Critical Ratios
\`\`\`
hip_to_shoulder_ratio = hip_width / shoulder_width
torso_proportion = vertical_torso_length / shoulder_width
waist_distance = distance(hip_midpoint, waist_point)
hip_to_waist_ratio = hip_width / waist_distance
\`\`\`

### Body Shape Classification
| Shape Type | Primary Criteria | Secondary Indicators |
|------------|------------------|----------------------|
| Hourglass | 0.9 ≤ hip_to_shoulder_ratio ≤ 1.1 | Defined waist, balanced proportions |
| Pear | hip_to_shoulder_ratio > 1.15 | Hips wider than shoulders |
| Inverted Triangle | hip_to_shoulder_ratio < 0.83 | Shoulders wider than hips |
| Rectangle | 0.9 ≤ hip_to_shoulder_ratio ≤ 1.1 | Minimal difference between measurements |
| Apple | hip_to_shoulder_ratio < 0.9 | Short vertical torso, low shoulder/hip differential |

IMPORTANT: Always classify as one of these five types. If calculations are inconclusive, select the closest match rather than returning "Undefined".

## 2. Skin Tone Analysis

### Handling Missing Data
If skin tone data is not provided:
- Use the directly provided \`skin_tone\` value from the input JSON
- NEVER return "Undefined" as a skin tone classification
- If no skin tone is specified, default to "Neutral" with Fitzpatrick 4 and a lower confidence score (70%)

### Primary Classification Sources
- Use \`fitzpatrick_classification\` or \`monk_classification\` from the tone detection response
- If both available, prioritize \`monk_classification\` for greater precision

### Tone Category Mapping

| Classification | Undertone Category | Characteristics |
|----------------|-------------------|----------------|
| Fitzpatrick 1-2, Monk 1-3 | Cool | Pink, red, or blue undertones |
| Fitzpatrick 3-4, Monk 4-7 | Neutral | Mixed undertones or balanced |
| Fitzpatrick 5-6, Monk 8-10 | Warm | Yellow, golden, or olive undertones |

## 4. Output Format

**Every thing you have to return in String**
Provide a detailed JSON response with the following structure:
json
{
  "bodyShape": "[hourglass/ apple/ pear/ rectangle/ invertedTriangle]",
  "skinTone": "[cool/ warm/ neutral]"
}


## 5. Critical Processing Rules

1. NEVER return "Undefined" for any classification - always select the best match from available options
2. If body_shape is directly provided in the input, use that value instead of calculating
3. If skin_tone is directly provided in the input, use that value instead of calculating
4. Always return a valid JSON object that matches the exact structure above
5. If measurements cannot be calculated, still provide the classification with a note in reasoningProcess
6. Handle all edge cases gracefully - the system must never fail to return a valid response
7. Ensure all numerical values are properly formatted as numbers, not strings
8. When direct values are used from input, set confidence to 95%`,
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