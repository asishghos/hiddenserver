import mongoose from 'mongoose';

// Reminder sub-schema for reusability
const reminderSchema = new mongoose.Schema({
    value: { 
        type: Number, 
        default: 1,
        min: [1, 'Reminder value must be at least 1']
    },
    type: { 
        type: String, 
        enum: ['Days before', 'Hours before', 'Weeks before', 'No reminders, I ❤️ FOMO'], 
        default: 'Days before' 
    },
    text: { 
        type: String, 
        default: '1 day before' 
    },
    isSent: { 
        type: Boolean, 
        default: false 
    }
}, { _id: false });

// Day-specific event sub-schema for multi-day events
const dayEventSchema = new mongoose.Schema({
    date: { 
        type: Date, 
        required: true 
    },
    eventName: { 
        type: String, 
        required: true,
        trim: true,
        maxlength: [100, 'Event name cannot exceed 100 characters']
    },
    eventTime: { 
        type: String, 
        required: true,
        trim: true
    },
    location: { 
        type: String, 
        required: true,
        trim: true,
        maxlength: [200, 'Location cannot exceed 200 characters']
    },
    description: { 
        type: String,
        trim: true,
        maxlength: [500, 'Description cannot exceed 500 characters']
    },
    reminder: {
        type: reminderSchema,
        default: () => ({})
    },
    // Day-specific styling/images if needed
    daySpecificImage: [{
        type: String,
        trim: true
    }]
}, { 
    _id: true, // Keep _id for individual day events
    timestamps: false 
});

// Main event schema
const eventSchema = new mongoose.Schema({
    userId: { 
        type: mongoose.Schema.Types.ObjectId, 
        ref: 'User', 
        required: true,
        index: true
    },
    
    // Basic event information (required for all events)
    title: { 
        type: String, 
        required: true,
        trim: true,
        maxlength: [100, 'Title cannot exceed 100 characters']
    },
    
    occasion: { 
        type: String, 
        required: true,
        trim: true,
        maxlength: [50, 'Occasion cannot exceed 50 characters']
    },
    
    // Date range (required for all events)
    startDate: { 
        type: Date, 
        required: true 
    },
    endDate: { 
        type: Date, 
        required: true,
        // validate: {
        //     validator: function(value) {
        //         return value >= this.startDate;
        //     },
        //     message: 'End date must be after or equal to start date'
        // }
    },
    
    // Event type
    isMultiDay: { 
        type: Boolean, 
        default: false,
        index: true
    },
    
    // Single-day event fields (only for non-multi-day events)
    singleDayDetails: {
        eventTime: { 
            type: String,
            required: function() { 
                return !this.isMultiDay; 
            },
            trim: true
        },
        location: { 
            type: String,
            required: function() { 
                return !this.isMultiDay; 
            },
            trim: true,
            maxlength: [100, 'Location cannot exceed 100 characters']
        },
        description: { 
            type: String,
            trim: true,
            maxlength: [500, 'Description cannot exceed 500 characters']
        },
        reminder: {
            type: reminderSchema,
            default: () => ({})
        }
    },
    
    // Multi-day event data (only for multi-day events)
    daySpecificData: {
        type: [dayEventSchema],
        validate: {
            validator: function(value) {
                if (this.isMultiDay) {
                    return value && value.length > 0;
                }
                return !value || value.length === 0;
            },
            message: 'Multi-day events must have at least one day-specific event'
        }
    },
    
    // Styling and media (common for both types)
    isStyled: { 
        type: Boolean, 
        default: false 
    },
    generatedImages: [{
        type: String,
        trim: true
    }],
    
    // Timezone handling
    timezone: { 
        type: String, 
        default: 'UTC',
        trim: true
    },
    
    // TTL field for automatic deletion
    expiresAt: { 
        type: Date, 
        default: function() {
            // Delete 1 days after end date (adjust as needed)
            return new Date(this.endDate.getTime() + (1 * 24 * 60 * 60 * 1000));
        }
    }
}, { 
    timestamps: true,
});

// Compound indexes for efficient queries
eventSchema.index({ userId: 1, startDate: 1 });
eventSchema.index({ userId: 1, endDate: 1 });
eventSchema.index({ userId: 1, isMultiDay: 1 });
eventSchema.index({ startDate: 1, endDate: 1 });

// TTL index for automatic cleanup
eventSchema.index({ expiresAt: 1 }, { expireAfterSeconds: 0 });

// Pre-save middleware for validation and data processing
eventSchema.pre('save', function(next) {
    // Ensure single-day events have proper date range
    if (!this.isMultiDay) {
        // For single-day events, endDate should equal startDate
        this.endDate = this.startDate;
        
        // Clear multi-day data if accidentally set
        this.daySpecificData = [];
    } else {
        // For multi-day events, clear single-day details
        this.singleDayDetails = {
            eventTime: undefined,
            location: undefined,
            description: undefined,
            reminder: undefined
        };
        
        // Validate that all day events fall within the date range
        // if (this.daySpecificData && this.daySpecificData.length > 0) {
        //     const invalidDates = this.daySpecificData.filter(dayEvent => 
        //         dayEvent.date < this.startDate || dayEvent.date > this.endDate
        //     );
            
        //     if (invalidDates.length > 0) {
        //         return next(new Error('All day events must fall within the event date range'));
        //     }
        // }
    }
    
    next();
});

export const Event = mongoose.model('Event', eventSchema);