import axios from "axios";
import fs from "fs";

async function checkByGPT(imagePath) {
  const messages = [
    {
      role: "system",
      content:
        "You are an assistant that checks if an uploaded image contains a full-body human photo.",
    },
    {
      role: "user",
      content: [
        {
          type: "text",
          text:
            "Please analyze the image and tell me if it shows a full-body image of a person. " +
            "Only return true if the image clearly shows the **entire human body from head to feet**. " +
            "If the image is of an object, animal, a random photo, or only a partial view of a human, return false. " +
            'Respond strictly with a JSON object in this format: { "isFullBody": true/false}.',
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

  const response = await axios.post(
    "https://api.openai.com/v1/chat/completions",
    {
      model: "gpt-4o",
      messages: messages,
      temperature: 0.3,
    },
    {
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      },
    }
  );

  const rawContent = response.data.choices[0].message.content;

  // Remove any markdown formatting before parsing
  const cleanContent = cleanJSONResponse(rawContent);

  return JSON.parse(cleanContent);
}

function cleanJSONResponse(rawContent) {
  return rawContent
    .replace(/^```json\s*/i, "")
    .replace(/^```\s*/i, "")
    .replace(/```$/, "")
    .trim();
}

export const imageCheck = async (req, res) => {
  const imagePath = req.file?.path;
  if (!imagePath) return res.status(400).json({ error: "No image uploaded" });

  try {
    const result = await checkByGPT(imagePath);
    console.log(result);
    res.json(result);
  } catch (error) {
    console.log(error);
    res.status(500).json({ error: error.message || "Processing failed" });
  } finally {
    if (fs.existsSync(imagePath)) fs.unlinkSync(imagePath);
  }
};
