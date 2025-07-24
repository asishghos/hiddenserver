import cron from 'node-cron';
import { Event } from './models/Event.js';
import { User } from './models/User.js';
import nodemailer from 'nodemailer';
import dotenv from "dotenv";

dotenv.config();

// Email transport (use environment variables for production)
const transporter = nodemailer.createTransport({
  service: 'Gmail',
  auth: {
    user: process.env.EMAIL_USER, // your gmail address
    pass: process.env.EMAIL_PASS, // app password
  },
});

// Run every minute
cron.schedule('* * * * *', async () => {
  const now = new Date();

  try {
    // Get all events with upcoming reminders
    const events = await Event.find({
      'reminders.isSent': false
    }).populate('userId');

    for (const event of events) {
      for (const reminder of event.reminders) {
        if (reminder.isSent) continue;

        const reminderTime = new Date(event.date.getTime() - reminder.timeBefore * 60000);

        if (reminderTime <= now && event.userId?.email) {
          // Send reminder email
          await transporter.sendMail({
            from: process.env.EMAIL_USER,
            to: event.userId.email,
            subject: `Reminder: ${event.name}`,
            text: `Hey! Your event "${event.name}" for "${event.occasion}" is coming up soon.`,
          });

          // Mark as sent
          reminder.isSent = true;
        }
      }

      await event.save();
    }

    console.log(`[${new Date().toISOString()}] Checked and sent reminders.`);
  } catch (err) {
    console.error('Reminder Cron Error:', err);
  }
});
