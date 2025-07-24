import sharp from "sharp";
import { Readable } from "stream";
import { writeFileSync } from "fs";
import { join } from "path";
import { randomUUID } from "crypto";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Save mask to file
async function toFile(stream, fileName = `${randomUUID()}.png`) {
  const filePath = join(__dirname, "..", "tmp", fileName);
  const chunks = [];

  for await (const chunk of stream) {
    chunks.push(chunk);
  }

  const buffer = Buffer.concat(chunks);
  writeFileSync(filePath, buffer);

  return {
    fileName,
    path: filePath,
    buffer,
    mimeType: "image/png",
  };
}

export async function createMask(imageBuffer) {
  try {
    const maskBuffer = await sharp({
      create: {
        width: 1024,
        height: 1024,
        channels: 4,
        background: { r: 255, g: 255, b: 255, alpha: 0 },
      },
    })
      .composite([
        {
          input: await sharp(imageBuffer)
            .resize(1024, 1024, { fit: "contain" })
            .ensureAlpha()
            .toBuffer(),
          blend: "over",
        },
      ])
      .png()
      .toBuffer();

    const stream = Readable.from(maskBuffer);
    return await toFile(stream);
  } catch (error) {
    console.error("Error creating mask:", error);
    throw error;
  }
}

// export default { createMask };
