import nodemailer from "nodemailer";

export const sendEmail = async ({ to, subject, text }) => {
    const transporter = nodemailer.createTransport({
        service: "gmail", // or use SendGrid, Mailgun etc
        auth: {
            user: process.env.EMAIL_USER,
            pass: process.env.EMAIL_PASS
        }
    });

    await transporter.sendMail({
        from: process.env.EMAIL_USER,
        to,
        subject,
        text
    });
};