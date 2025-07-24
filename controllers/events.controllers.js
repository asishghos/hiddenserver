import { Event } from "../models/events.models.js";

const formatEventResponse = (event) => {
  return {
    id: event._id,
    userId: event.userId,
    title: event.title,
    occasion: event.occasion,
    startDate: event.startDate,
    endDate: event.endDate,
    isMultiDay: event.isMultiDay,
    timezone: event.timezone,
    isStyled: event.isStyled,
    generatedImages: event.generatedImages,
    singleDayDetails: event.singleDayDetails || null,
    daySpecificData:
      event.daySpecificData?.map((day) => ({
        id: day._id,
        date: day.date,
        eventName: day.eventName,
        eventTime: day.eventTime,
        location: day.location,
        description: day.description,
        reminder: day.reminder,
        daySpecificImage: day.daySpecificImage || null,
      })) || [],
  };
};

export const addEvent = async (req, res) => {
  const user = req.user;
  if (!user) return res.status(401).json({ message: "Unauthorized" });

  try {
    const userId = user._id;
    const {
      title,
      occasion,
      startDate,
      endDate,
      isMultiDay,
      eventName,
      eventTime,
      location,
      description,
      reminder,
      reminderValue,
      reminderType,
      daySpecificData,
      timezone,
    } = req.body;

    const eventData = {
      userId,
      title,
      occasion,
      startDate: new Date(startDate),
      endDate: new Date(endDate),
      isMultiDay,
      timezone: timezone || "UTC",
    };

    if (isMultiDay) {
      eventData.daySpecificData = Object.entries(daySpecificData).map(
        ([dateStr, data]) => ({
          date: new Date(dateStr),
          eventName: data.eventName,
          eventTime: data.eventTime,
          location: data.location,
          description: data.description,
          reminder: {
            value: data.reminder?.value || 1,
            type: data.reminder?.type || "Days before",
            text: data.reminder?.text || "1 day before",
            isSent: false,
          },
        })
      );
    } else {
      eventData.singleDayDetails = {
        eventTime,
        location,
        description,
        reminder: {
          value: reminderValue || 1,
          type: reminderType || "Days before",
          text: reminder || "1 day before",
          isSent: false,
        },
      };
    }

    const event = new Event(eventData);
    await event.save();

    res.status(201).json({
      message: "Event created successfully",
      event: formatEventResponse(event),
    });
  } catch (error) {
    console.error("Error creating event:", error);
    res.status(400).json({ error: error.message });
  }
};

export const getEventDetails = async (req, res) => {
  try {
    const { id } = req.params;
    if (!id) return res.status(400).json({ message: "Event ID is required" });

    const event = await Event.findById(id);
    if (!event) return res.status(404).json({ message: "Event not found" });

    if (event.userId.toString() !== req.user._id.toString()) {
      return res
        .status(403)
        .json({ message: "Unauthorized to view this event" });
    }

    return res.status(200).json({
      message: "Event details fetched successfully",
      event: formatEventResponse(event),
    });
  } catch (error) {
    console.error("Error fetching event details:", error.message);
    return res
      .status(500)
      .json({ message: "Error fetching event details", error: error.message });
  }
};

export const getUpcomingEvents = async (req, res) => {
  try {
    const user = req.user;
    if (!user) return res.status(401).json({ message: "Unauthorized" });

    const currentDate = new Date();

    const upcomingEvents = await Event.find({
      userId: user._id,
    //   startDate: { $gte: currentDate },
    }).sort({ startDate: 1 });

    const formatted = upcomingEvents.map(formatEventResponse);
    // console.log("Formatted Upcoming Events:", formatted);
    return res.status(200).json({
      message: "Upcoming events fetched successfully",
      events: formatted,
    });
  } catch (error) {
    console.error("Error fetching upcoming events:", error.message);
    return res
      .status(500)
      .json({
        message: "Error fetching upcoming events",
        error: error.message,
      });
  }
};

export const updateEvent = async (req, res) => {
  const { id, dayEventId } = req.params;
  const user = req.user;
  if (!user) return res.status(401).json({ message: "Unauthorized" });

  try {
    const event = await Event.findById(id);
    if (!event) return res.status(404).json({ message: "Event not found" });

    if (event.userId.toString() !== user._id.toString()) {
      return res
        .status(403)
        .json({ message: "Unauthorized to update this event" });
    }

    if (dayEventId) {
      if (!event.isMultiDay) {
        return res
          .status(400)
          .json({ message: "This is not a multi-day event" });
      }

      const dayEvent = event.daySpecificData.id(dayEventId);
      if (!dayEvent) {
        return res.status(404).json({ message: "Day event not found" });
      }

      const {
        eventName,
        time,
        location,
        description,
        // daySpecificImage,
        reminder,
        reminderValue,
        reminderType,
      } = req.body;

      if (eventName) dayEvent.eventName = eventName;
      if (time) dayEvent.eventTime = time;
      if (location) dayEvent.location = location;
      if (description !== undefined) dayEvent.description = description;
      // if (daySpecificImage !== undefined)
      //   dayEvent.daySpecificImage = daySpecificImage;

      if (reminder || reminderValue || reminderType) {
        dayEvent.reminder = {
          value: reminderValue || dayEvent.reminder.value,
          type: reminderType || dayEvent.reminder.type,
          text: reminder || dayEvent.reminder.text,
          isSent: false,
        };
      }

      await event.save();

      return res.status(200).json({
        message: "Day event updated successfully",
        event: formatEventResponse(event),
      });
    }

    const {
      title,
      occasion,
      startDate,
      endDate,
      isMultiDay,
      time,
      location,
      description,
      reminder,
      reminderValue,
      reminderType,
      daySpecificData,
      isStyled,
      generatedImages,
      timezone,
    } = req.body;

    const updatedFields = {
      title: title || event.title,
      occasion: occasion || event.occasion,
      startDate: startDate ? new Date(startDate) : event.startDate,
      endDate: endDate ? new Date(endDate) : event.endDate,
      isMultiDay: isMultiDay !== undefined ? isMultiDay : event.isMultiDay,
      isStyled: isStyled !== undefined ? isStyled : event.isStyled,
      generatedImages: generatedImages || event.generatedImages,
      timezone: timezone || event.timezone,
    };

    if (updatedFields.isMultiDay) {
      if (daySpecificData) {
        updatedFields.daySpecificData = Object.entries(daySpecificData).map(
          ([dateStr, data]) => ({
            date: new Date(dateStr),
            eventName: data.event,
            eventTime: data.time,
            location: data.location,
            description: data.description,
            reminder: {
              value: data.reminderValue || 1,
              type: data.reminderType || "Days before",
              text: data.reminder || "1 day before",
              isSent: false,
            },
          })
        );
      }
      updatedFields.singleDayDetails = undefined;
    } else {
      updatedFields.singleDayDetails = {
        eventTime: time || event.singleDayDetails?.eventTime,
        location: location || event.singleDayDetails?.location,
        description: description || event.singleDayDetails?.description,
        reminder: {
          value: reminderValue || event.singleDayDetails?.reminder?.value || 1,
          type:
            reminderType ||
            event.singleDayDetails?.reminder?.type ||
            "Days before",
          text:
            reminder ||
            event.singleDayDetails?.reminder?.text ||
            "1 day before",
          isSent: false,
        },
      };
      updatedFields.daySpecificData = [];
    }

    const updatedEvent = await Event.findByIdAndUpdate(id, updatedFields, {
      new: true,
      runValidators: true,
    });

    return res.status(200).json({
      message: "Event updated successfully",
      event: formatEventResponse(updatedEvent),
    });
  } catch (error) {
    console.error("Error updating event:", error);
    res
      .status(500)
      .json({ message: "Error updating event", error: error.message });
  }
};

export const deleteEvent = async (req, res) => {
  const { id, dayEventId } = req.params;
  const user = req.user;
  if (!user) return res.status(401).json({ message: "Unauthorized" });

  try {
    const event = await Event.findById(id);
    if (!event) {
      return res.status(404).json({ message: "Event not found" });
    }

    if (event.userId.toString() !== user._id.toString()) {
      return res
        .status(403)
        .json({ message: "Unauthorized to delete this event" });
    }

    if (dayEventId) {
      if (!event.isMultiDay) {
        return res
          .status(400)
          .json({ message: "This is not a multi-day event" });
      }

      const dayEvent = event.daySpecificData.id(dayEventId);
      if (!dayEvent) {
        return res.status(404).json({ message: "Day event not found" });
      }

      if (event.daySpecificData.length === 1) {
        return res.status(400).json({
          message:
            "Cannot delete the last day event. Delete the entire multi-day event instead.",
        });
      }

      event.daySpecificData.pull(dayEventId);
      await event.save();

      return res.status(200).json({
        message: "Day event deleted successfully",
        event: formatEventResponse(event),
        remainingDayEvents: event.daySpecificData.length,
      });
    }

    await Event.findByIdAndDelete(id);

    return res.status(200).json({
      message: "Event deleted successfully",
      eventId: id,
    });
  } catch (error) {
    console.error("Error deleting event:", error);
    return res
      .status(500)
      .json({ message: "Error deleting event", error: error.message });
  }
};

export const getMultiDayEventCollection = async (req, res) => {
  try {
    const { id } = req.params;
    const user = req.user;
    if (!user) return res.status(401).json({ message: "Unauthorized" });
    if (!id)
      return res
        .status(400)
        .json({ message: "Multi-day event ID is required" });

    const multiDayEvent = await Event.findById(id);
    if (!multiDayEvent)
      return res.status(404).json({ message: "Multi-day event not found" });
    if (multiDayEvent.userId.toString() !== user._id.toString()) {
      return res
        .status(403)
        .json({ message: "Unauthorized to view this event" });
    }
    if (!multiDayEvent.isMultiDay) {
      return res.status(400).json({ message: "This is not a multi-day event" });
    }

    const dayEvents = multiDayEvent.daySpecificData
      .map((dayEvent) => ({
        id: dayEvent._id,
        parentEventId: multiDayEvent._id,
        parentTitle: multiDayEvent.title,
        parentOccasion: multiDayEvent.occasion,
        date: dayEvent.date,
        eventName: dayEvent.eventName,
        eventTime: dayEvent.eventTime,
        location: dayEvent.location,
        description: dayEvent.description,
        reminder: dayEvent.reminder,
        daySpecificImage: dayEvent.daySpecificImage,
        isStyled: multiDayEvent.isStyled,
        generatedImages: multiDayEvent.generatedImages,
      }))
      .sort((a, b) => new Date(a.date) - new Date(b.date));

    return res.status(200).json({
      message: "Multi-day event collection fetched successfully",
      parentEvent: formatEventResponse(multiDayEvent),
      dayEvents,
      totalDayEvents: dayEvents.length,
    });
  } catch (error) {
    console.error("Error fetching multi-day event collection:", error.message);
    return res.status(500).json({
      message: "Error fetching multi-day event collection",
      error: error.message,
    });
  }
};

export const getEventReminders = async (req, res) => {
  try {
    const user = req.user;
    if (!user) return res.status(401).json({ message: "Unauthorized" });

    const currentDate = new Date();
    const events = await Event.find({
      userId: user._id,
      startDate: { $gte: currentDate },
    });

    const reminders = [];

    events.forEach((event) => {
      if (event.isMultiDay) {
        event.daySpecificData.forEach((day) => {
          if (!day.reminder?.isSent) {
            reminders.push({
              eventId: event._id,
              dayEventId: day._id,
              eventName: day.eventName,
              date: day.date,
              reminder: day.reminder,
              isMultiDay: true,
            });
          }
        });
      } else if (
        event.singleDayDetails?.reminder &&
        !event.singleDayDetails.reminder.isSent
      ) {
        reminders.push({
          eventId: event._id,
          eventName: event.title,
          date: event.startDate,
          reminder: event.singleDayDetails.reminder,
          isMultiDay: false,
        });
      }
    });

    return res.status(200).json({
      message: "Event reminders fetched successfully",
      reminders,
    });
  } catch (error) {
    console.error("Error fetching event reminders:", error.message);
    return res
      .status(500)
      .json({
        message: "Error fetching event reminders",
        error: error.message,
      });
  }
};

// export const markReminderAsSent = async (req, res) => {
//     try {
//         const { eventId } = req.params;
//         const { dayDate } = req.body; // For multi-day events

//         const event = await Event.findById(eventId);

//         if (!event) {
//             return res.status(404).json({ message: 'Event not found' });
//         }

//         if (event.userId.toString() !== req.user._id.toString()) {
//             return res.status(403).json({ message: 'Unauthorized' });
//         }

//         if (event.isMultiDay && dayDate) {
//             // Mark specific day reminder as sent
//             const dayIndex = event.daySpecificData.findIndex(
//                 day => day.date.toISOString().split('T')[0] === new Date(dayDate).toISOString().split('T')[0]
//             );

//             if (dayIndex !== -1) {
//                 event.daySpecificData[dayIndex].reminder.isSent = true;
//             }
//         } else {
//             // Mark single-day event reminder as sent
//             event.reminder.isSent = true;
//         }

//         await event.save();

//         return res.status(200).json({
//             message: 'Reminder marked as sent',
//             event
//         });
//     } catch (error) {
//         console.error("Error marking reminder as sent:", error.message);
//         return res.status(500).json({ message: 'Error updating reminder status', error: error.message });
//     }
// };

// function buildOutfitPrompt(occasion, dayTime, description, userBodyInfo) {
//     const bodyShape = userBodyInfo?.bodyShapeAnalysis.bodyShape.classification || "not specified";
//     const skinTone = userBodyInfo?.bodyShapeAnalysis.skinTone.toneCategory || "not specified";
//     const gender = userBodyInfo?.bodyShapeAnalysis.gender.classification || "not specified";
//     const desc = description?.trim()
//         ? `Additional styling notes: ${description.trim()}.`
//         : "Focus on classic, versatile styling suitable for everyday wear.";

//     return `
//         You are a professional fashion stylist and visual designer. Your task is to generate a **photorealistic, magazine-quality outfit image** based on the user's profile and occasion. Design a full outfit that is cohesive, stylish, and wearable.

//         👤 USER PROFILE:
//         - Body shape: ${bodyShape}
//         - Skin tone: ${skinTone}
//         - Gender: ${gender}

//         📅 CONTEXT:
//         - Occasion: ${occasion || "everyday wear"}
//         - Time of day: ${dayTime || "anytime"}
//         - ${desc}

//         👗 OUTFIT DESIGN:
//         - Include: top, bottom, shoes, and accessories (bag, jewelry, scarf, etc.)
//         - Color and fabric choices should suit skin tone and occasion
//         - Make it season-appropriate and culturally sensitive
//         - The overall outfit must be harmonious, flattering, and elegant
//         - Ensure each piece fits well and complements body shape

//         🖼️ VISUAL STYLE:
//         - High-resolution, clear fabric details
//         - Natural, soft lighting with gentle shadows
//         - Neutral background (light gray, beige, or off-white)
//         - Position outfit clearly to showcase all components

//         ✨ STYLE DIRECTION:
//         - Blend timeless elegance with a modern twist
//         - Colors should be balanced and fashion-forward
//         - Add tasteful styling elements that elevate the look
//         - Prioritize comfort, functionality, and real-world wearability

//         Final goal: A visually stunning, realistic outfit that aligns with user traits and the specific occasion. The result should look like it belongs on the cover of a fashion magazine.
//     `.replace(/\s+/g, ' ').trim();
// }

// async function generateOutfitAsPerRequirements(occasion, dayTime, description, userBodyInfo) {
//     try {
//         const prompt = buildOutfitPrompt(occasion, dayTime, description, userBodyInfo);

//         const response = await ai.models.generateContent({
//             model: "gemini-2.0-flash-preview-image-generation",
//             contents: prompt,
//             config: {
//                 responseModalities: [Modality.TEXT, Modality.IMAGE],
//             },
//         });

//         const result = {
//             textDescription: "",
//             imageUrl: "",
//             status: "success",
//         };

//         for (const part of response.candidates[0].content.parts) {
//             if (part.text) {
//                 result.textDescription = part.text;
//             } else if (part.inlineData) {
//                 const imageDataB64 = part.inlineData.data;
//                 const buffer = Buffer.from(imageDataB64, "base64");

//                 // Ensure folder exists
//                 const imagesDir = path.join("static", "images");
//                 fs.mkdirSync(imagesDir, { recursive: true });

//                 const filename = `outfit_${Date.now()}.png`;
//                 const filepath = path.join(imagesDir, filename);
//                 fs.writeFileSync(filepath, buffer);

//                 // Set accessible image URL
//                 result.imageUrl = `/static/images/${filename}`;
//             }
//         }

//         return {
//             success: true,
//             data: result,
//         };

//     } catch (error) {
//         console.error("Generation Error:", error);
//         return {
//             success: false,
//             error: error.message,
//         };
//     }
// }

// export const styleForEvent = async (req, res) => {
//     try {
//         const event = await Event.findById(req.params.id);
//         if (!event) {
//             return res.status(404).json({ message: 'Event not found' });
//         }

//         const userId = req.user._id;
//         // Check if the user is authorized to style this event
//         if(event.userId.toString() !== userId.toString()) {
//             return res.status(403).json({ message: 'Unauthorized to style this event' });
//         }

//         const userBodyInfo = req.user.userBodyInfo;

//         // fetch user body data here to give more personalized outfit

//         if (!event.isStyled || !event.generatedImages || event.generatedImages.length === 0) {
//             const genResult = await generateOutfitAsPerRequirements(
//                 event.occasion,
//                 event.dayTime,
//                 event.description,
//                 userBodyInfo
//             );

//             // console.log("GEN RESULT:", genResult);

//             if (!genResult.success || !genResult.data?.imageUrl) {
//                 return res.status(500).json({ message: 'Error generating outfit' });
//             }

//             // Save the image as base64 or link to image hosting
//             event.isStyled = true;
//             event.generatedImages = [genResult.data.imageUrl];
//             await event.save();

//             return res.status(200).json({
//                 message: 'Event styled successfully',
//                 description: genResult.data.textDescription,
//                 images: event.generatedImages,
//             });
//         }

//         return res.status(200).json({
//             message: 'Event already styled',
//             images: event.generatedImages,
//         });
//     } catch (error) {
//         console.error("Error styling event:", error.message);
//         return res.status(500).json({ message: 'Error styling event', error: error.message });
//     }
// };
