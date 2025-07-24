import express from 'express';
import { ImageResponse } from '@vercel/og';
import fetch from 'node-fetch'; // For fetching remote logos or images

const app = express();
app.use(express.json());

app.post('/generate-image', async (req, res) => {
  const {
    title = 'My App Title',
    subtitle = 'Best app of 2025',
    description = 'Download now and enjoy amazing features!',
    imageUrl = 'https://via.placeholder.com/600x300',
    brandLogo = 'https://via.placeholder.com/100'
  } = req.body;

  try {
    const imageResponse = new ImageResponse(
      (
        <div
          style={{
            background: '#fdfdfd',
            height: '100%',
            width: '100%',
            display: 'flex',
            flexDirection: 'column',
            padding: '40px',
            fontFamily: 'Arial',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}
        >
          <img src={brandLogo} width="100" style={{ borderRadius: '50%' }} />
          <h1 style={{ fontSize: 48, margin: '20px 0', color: '#111' }}>{title}</h1>
          <h2 style={{ fontSize: 28, margin: '10px 0', color: '#555' }}>{subtitle}</h2>
          <p style={{ fontSize: 24, color: '#666' }}>{description}</p>
          <img src={imageUrl} width="600" style={{ borderRadius: 12 }} />
        </div>
      ),
      {
        width: 1200,
        height: 630,
      }
    );

    // Convert it to base64
    const arrayBuffer = await imageResponse.arrayBuffer();
    const base64 = Buffer.from(arrayBuffer).toString('base64');

    res.json({
      base64: `data:image/png;base64,${base64}`,
      message: 'Image generated successfully!',
    });
  } catch (err) {
    console.error('Error:', err);
    res.status(500).json({ error: 'Failed to generate image' });
  }
});