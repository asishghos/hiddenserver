import cv2
import numpy as np
from sklearn.cluster import KMeans
import requests
import base64
import json
import os
import sys
from PIL import Image
import io
import colorsys

def extract_skin_regions_using_yolo(image, keypoints):
    """
    Extract skin regions based on YOLO keypoints.
    This function uses body keypoints to identify face and exposed skin areas.
    
    Args:
        image: RGB image array
        keypoints: YOLO keypoints dictionary
    
    Returns:
        List of regions (x, y, w, h) containing skin
    """
    regions = []
    
    # Extract face region based on keypoints
    if 'keypoints' not in keypoints:
        return regions
    
    kp = keypoints['keypoints']
    
    # Extract face region if nose and eyes are detected
    face_keypoints = []
    for point_name in ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']:
        if point_name in kp:
            face_keypoints.append(kp[point_name])
    
    if len(face_keypoints) >= 3:  # At least nose and eyes
        x_coords = [p[0] for p in face_keypoints]
        y_coords = [p[1] for p in face_keypoints]
        
        # Calculate bounding box with some margin
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Add margins
        width = max_x - min_x
        height = max_y - min_y
        
        # Expanded face region
        face_x = max(0, int(min_x - width * 0.3))
        face_y = max(0, int(min_y - height * 0.3))
        face_w = int(width * 1.6)
        face_h = int(height * 1.8)
        
        # Make sure we don't exceed image dimensions
        h, w = image.shape[:2]
        face_w = min(face_w, w - face_x)
        face_h = min(face_h, h - face_y)
        
        regions.append((face_x, face_y, face_w, face_h))
    
    # Additionally, extract forearm areas if available
    arm_keypoints = []
    for point_name in ['left_wrist', 'right_wrist', 'left_elbow', 'right_elbow']:
        if point_name in kp:
            arm_keypoints.append(kp[point_name])
    
    # Process arm regions if enough keypoints are available
    if len(arm_keypoints) >= 2:
        for i in range(len(arm_keypoints) - 1):
            x1, y1 = arm_keypoints[i]
            x2, y2 = arm_keypoints[i+1]
            
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)
            
            # Add margins
            width = max(max_x - min_x, 30)
            height = max(max_y - min_y, 30)
            
            arm_x = max(0, int(min_x - width * 0.2))
            arm_y = max(0, int(min_y - height * 0.2))
            arm_w = int(width * 1.4)
            arm_h = int(height * 1.4)
            
            # Check bounds
            h, w = image.shape[:2]
            arm_w = min(arm_w, w - arm_x)
            arm_h = min(arm_h, h - arm_y)
            
            regions.append((arm_x, arm_y, arm_w, arm_h))
    
    return regions

def remove_non_skin_regions(image, region):
    """
    Attempt to filter out non-skin pixels from the region using multiple color spaces.
    
    Args:
        image: RGB image array
        region: (x, y, w, h) tuple
    
    Returns:
        Filtered image with likely skin pixels
    """
    x, y, w, h = region
    roi = image[y:y+h, x:x+w].copy()
    
    # Convert to HSV color space for better skin detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    
    # Use multiple color spaces for more robust detection
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_RGB2YCrCb)
    
    # Define range for skin tone in HSV - WIDER RANGE
    lower_skin_hsv = np.array([0, 15, 30])
    upper_skin_hsv = np.array([50, 255, 255])
    
    # Define range for skin in YCrCb color space
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([255, 173, 127], np.uint8)
    
    # Create masks
    mask_hsv = cv2.inRange(hsv, lower_skin_hsv, upper_skin_hsv)
    mask_ycrcb = cv2.inRange(ycrcb, min_YCrCb, max_YCrCb)
    
    # Combine the masks
    mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)
    
    # Apply morphological operations to improve mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply mask to original region
    skin = cv2.bitwise_and(roi, roi, mask=mask)
    
    # If we have very few skin pixels, use a more permissive approach as fallback
    if np.count_nonzero(mask) < 100:
        # Just use the original ROI with minimal filtering
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)  # Remove very dark pixels
        skin = cv2.bitwise_and(roi, roi, mask=mask)
    
    return skin

def get_dominant_colors(image, n_colors=3):
    """
    Extract dominant colors from an image.
    
    Args:
        image: RGB image array
        n_colors: Number of dominant colors to extract
    
    Returns:
        List of (r, g, b) tuples of dominant colors
    """
    pixels = image.reshape(-1, 3)
    
    # Filter out black pixels (likely background)
    pixels = pixels[~np.all(pixels == 0, axis=1)]
    
    if len(pixels) == 0:
        return [(0, 0, 0)]  # Return black if no valid pixels
    
    # If very few pixels, adjust n_colors to prevent KMeans failure
    if len(pixels) < n_colors:
        n_colors = max(1, len(pixels) // 2)
    
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    colors = kmeans.cluster_centers_
    
    labels = kmeans.labels_
    counts = np.bincount(labels)
    
    colors_with_counts = [(colors[i], counts[i]) for i in range(len(counts))]
    colors_with_counts.sort(key=lambda x: x[1], reverse=True)
    
    return [(int(c[0]), int(c[1]), int(c[2])) for c, _ in colors_with_counts]

def classify_undertone(rgb_color):
    """
    Determine the undertone of a skin color using improved logic with enhanced
    detection for cool undertones, especially in fair skin.
    
    Args:
        rgb_color: (r, g, b) tuple
    
    Returns:
        String describing the undertone (cool, neutral, warm, olive)
    """
    r, g, b = rgb_color
    
    # Convert RGB to HSV for better hue analysis
    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    
    # Convert to degrees for easier comparison (0-360)
    h_degrees = h * 360
    
    # Calculate color ratios
    if (r + g + b) > 0:
        r_ratio = r / (r + g + b)
        g_ratio = g / (r + g + b)
        b_ratio = b / (r + g + b)
    else:
        r_ratio = g_ratio = b_ratio = 0
    
    # Red to blue ratio helps determine warm vs cool
    rb_ratio = r / b if b > 0 else 10
    # Blue to red ratio for cool undertones
    br_ratio = b / r if r > 0 else 0
    
    # Green presence helps identify olive undertones
    green_presence = g_ratio - (r_ratio + b_ratio) / 2
    
    # Calculate LAB color space values (approximation)
    var_r = r/255.0
    var_g = g/255.0
    var_b = b/255.0

    # Convert RGB to XYZ
    var_r = (var_r > 0.04045) and ((var_r + 0.055) / 1.055) ** 2.4 or var_r / 12.92
    var_g = (var_g > 0.04045) and ((var_g + 0.055) / 1.055) ** 2.4 or var_g / 12.92
    var_b = (var_b > 0.04045) and ((var_b + 0.055) / 1.055) ** 2.4 or var_b / 12.92

    var_r *= 100
    var_g *= 100
    var_b *= 100

    # Observer = 2°, Illuminant = D65
    X = var_r * 0.4124 + var_g * 0.3576 + var_b * 0.1805
    Y = var_r * 0.2126 + var_g * 0.7152 + var_b * 0.0722
    Z = var_r * 0.0193 + var_g * 0.1192 + var_b * 0.9505

    # Convert XYZ to LAB
    X /= 95.047
    Y /= 100.000
    Z /= 108.883

    X = (X > 0.008856) and X ** (1/3) or (7.787 * X) + (16 / 116)
    Y = (Y > 0.008856) and Y ** (1/3) or (7.787 * Y) + (16 / 116)
    Z = (Z > 0.008856) and Z ** (1/3) or (7.787 * Z) + (16 / 116)

    L = (116 * Y) - 16
    a = 500 * (X - Y)  # Positive a = red, negative a = green
    b_val = 200 * (Y - Z)  # Positive b = yellow, negative b = blue
    
    # Enhanced cool undertone detection for fair skin
    # Check if the skin is fair to medium (higher luminance)
    is_fair_skin = L > 70
    
    # Check for blue or pink tinge common in cool undertones
    blue_presence = b_ratio > 0.31  # Higher blue component
    cool_hue = (h_degrees >= 270 or h_degrees <= 30)  # Purple-blue to pinkish range
    
    # Enhanced detection for cool undertones in fair skin
    if is_fair_skin and (b_val < -2 or br_ratio > 0.85 or blue_presence):
        return "cool"
    
    # Rest of the undertone classification logic
    if green_presence > 0.02 and g_ratio > 0.33:
        return "olive"
    elif a < 0 and b_val < 0:  # Greenish-blue bias
        return "cool"
    elif b_val < -4:  # Stronger blue component
        return "cool"
    elif a < 8 and abs(b_val) < 6:  # Balanced color
        return "neutral"
    elif a > 8 and b_val > 0:  # Red-yellow bias
        return "warm"
    elif a > 0 and b_val < 0:  # Red-blue mix
        if abs(b_val) > 4:  # If blue component is significant
            return "cool"
        else:
            return "neutral-cool"
    elif a < 0 and b_val > 0:  # Green-yellow mix
        return "neutral-olive"
    else:
        # Default to neutral if none of the specific conditions match
        return "neutral"

def classify_skin_tone(rgb_color):
    """
    Map RGB color to a skin tone classification using the Fitzpatrick scale.
    
    Args:
        rgb_color: (r, g, b) tuple
    
    Returns:
        String describing the skin tone and its Fitzpatrick Type
    """
    r, g, b = rgb_color
    
    luminance = 0.299*r + 0.587*g + 0.114*b
    
    # Map to Fitzpatrick Scale with descriptions
    if luminance > 200:
        return "Very Fair (Fitzpatrick Type I)", 1
    elif luminance > 180:
        return "Fair (Fitzpatrick Type II)", 2
    elif luminance > 160:
        return "Light/Medium (Fitzpatrick Type III)", 3
    elif luminance > 120:
        return "Moderate Brown/Olive (Fitzpatrick Type IV)", 4
    elif luminance > 80:
        return "Brown (Fitzpatrick Type V)", 5
    else:
        return "Dark Brown/Black (Fitzpatrick Type VI)", 6

def classify_monk_skin_tone(rgb_color):
    """
    Map RGB color to the Monk Skin Tone Scale (10-point scale).
    
    Args:
        rgb_color: (r, g, b) tuple
    
    Returns:
        Tuple of (string description, numeric value 1-10)
    """
    r, g, b = rgb_color
    
    luminance = 0.299*r + 0.587*g + 0.114*b
    
    # Determine undertone
    undertone = classify_undertone(rgb_color)
    
    if luminance > 220:
        return f"Lightest (Monk Scale 1) with {undertone} undertones", 1
    elif luminance > 200:
        return f"Very Light (Monk Scale 2) with {undertone} undertones", 2
    elif luminance > 180:
        return f"Light (Monk Scale 3) with {undertone} undertones", 3
    elif luminance > 160:
        return f"Light Medium (Monk Scale 4) with {undertone} undertones", 4
    elif luminance > 140:
        return f"Medium (Monk Scale 5) with {undertone} undertones", 5
    elif luminance > 120:
        return f"Medium Deep (Monk Scale 6) with {undertone} undertones", 6
    elif luminance > 100:
        return f"Deep (Monk Scale 7) with {undertone} undertones", 7
    elif luminance > 80:
        return f"Deep Dark (Monk Scale 8) with {undertone} undertones", 8
    elif luminance > 60:
        return f"Very Dark (Monk Scale 9) with {undertone} undertones", 9
    else:
        return f"Darkest (Monk Scale 10) with {undertone} undertones", 10

def detect_skin_tone_from_image(image, keypoints=None):
    """
    Detect skin tone from an image using YOLO keypoints or fallback to whole image analysis.
    
    Args:
        image: RGB image array
        keypoints: Optional dictionary of YOLO keypoints
    
    Returns:
        Dictionary with skin tone information
    """
    if keypoints is None:
        keypoints = {"keypoints": {}}
    
    skin_regions = extract_skin_regions_using_yolo(image, keypoints)
    
    # If no regions detected or very few keypoints, use multiple fallback regions
    if not skin_regions or len(keypoints.get("keypoints", {})) < 3:
        h, w = image.shape[:2]
        
        # Add multiple regions to increase chances of finding skin
        # Center region
        center_x = w // 4
        center_y = h // 4
        center_w = w // 2
        center_h = h // 2
        skin_regions.append((center_x, center_y, center_w, center_h))
        
        # Face region (upper center of the image)
        face_x = w // 3
        face_y = h // 6
        face_w = w // 3
        face_h = h // 3
        skin_regions.append((face_x, face_y, face_w, face_h))
        
        # If still no regions, use the entire image as last resort
        if not skin_regions:
            skin_regions = [(0, 0, w, h)]
    
    all_skin_pixels = []
    
    # Process each region to find skin pixels
    for region in skin_regions:
        skin_image = remove_non_skin_regions(image, region)
        
        # Extract non-black pixels (potential skin)
        non_black = skin_image[np.where((skin_image != [0,0,0]).all(axis=2))]
        
        if len(non_black) > 0:
            all_skin_pixels.extend(non_black)
    
    # If still no skin pixels, use a more aggressive approach
    if not all_skin_pixels:
        print("No skin pixels detected with normal filtering, using fallback method")
        # Use a more permissive approach: sample center regions directly
        h, w = image.shape[:2]
        
        # Sample the center quarter of the image
        center_x = w // 4
        center_y = h // 4
        center_w = w // 2
        center_h = h // 2
        
        center_region = image[center_y:center_y+center_h, center_x:center_x+center_w]
        
        # Simple filtering: remove very dark and very light pixels
        center_pixels = center_region.reshape(-1, 3)
        brightness = np.sum(center_pixels, axis=1)
        
        # Keep pixels with reasonable brightness (not too dark, not too bright)
        moderate_brightness = center_pixels[(brightness > 150) & (brightness < 700)]
        
        if len(moderate_brightness) > 0:
            all_skin_pixels = moderate_brightness
        else:
            # Last resort: just use central pixels
            all_skin_pixels = center_pixels
    
    if not all_skin_pixels:
        return {"error": "No skin pixels detected after all filtering attempts"}
    
    all_skin_pixels = np.array(all_skin_pixels)
    
    # Get dominant colors from the skin pixels
    try:
        dominant_colors = get_dominant_colors(all_skin_pixels, n_colors=min(3, len(all_skin_pixels) // 100 + 1))
        
        main_color = dominant_colors[0]
        fitzpatrick_result, fitzpatrick_value = classify_skin_tone(main_color)
        monk_result, monk_value = classify_monk_skin_tone(main_color)
        
        # Add standalone undertone classification
        undertone = classify_undertone(main_color)
        
        # Analyze each dominant color for more comprehensive results
        undertone_analysis = []
        for color in dominant_colors[:3]:  # Analyze up to 3 dominant colors
            if len(undertone_analysis) < 3:  # Limit to 3 results
                undertone_analysis.append({
                    "color_rgb": color,
                    "undertone": classify_undertone(color)
                })
        
        return {
            "dominant_colors": dominant_colors,
            "main_color_rgb": main_color,
            "fitzpatrick_classification": fitzpatrick_result,
            "fitzpatrick_value": fitzpatrick_value,
            "monk_classification": monk_result,
            "monk_value": monk_value,
            "undertone": undertone,
            "undertone_analysis": undertone_analysis
        }
    except Exception as e:
        return {
            "error": f"Error in color analysis: {str(e)}",
            "pixel_count": len(all_skin_pixels),
            "sample_colors": all_skin_pixels[:5].tolist() if len(all_skin_pixels) >= 5 else all_skin_pixels.tolist()
        }

def process_base64_image(base64_data):
    """
    Process an image from base64 encoding
    
    Args:
        base64_data: Base64 encoded image data
    
    Returns:
        RGB image array
    """
    try:
        # Decode base64 data to image
        # Handle potential padding issues
        base64_data = base64_data.strip()
        # Add padding if necessary
        padding = 4 - (len(base64_data) % 4) if len(base64_data) % 4 else 0
        base64_data += '=' * padding
        
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array and ensure RGB format
        image_array = np.array(image)
        
        # Check if image is RGBA and convert to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            # Use PIL for better RGBA to RGB conversion
            image = image.convert('RGB')
            image_array = np.array(image)
        
        # If image is grayscale, convert to RGB
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif len(image_array.shape) == 3 and image_array.shape[2] == 1:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # Already RGB from PIL
            pass
        
        return image_array
        
    except Exception as e:
        print(f"Error in processing base64 image: {str(e)}")
        raise e

def parse_yolo_keypoints(keypoints_text):
    """
    Parse YOLO keypoint text into a structured format.
    
    Args:
        keypoints_text: String with YOLO keypoint information
    
    Returns:
        Dictionary with keypoint information
    """
    keypoint_dict = {}
    
    # Define keypoint names based on YOLO standard
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    try:
        lines = keypoints_text.strip().split('\n')
        for line in lines:
            if "Keypoint" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    index_part = parts[0].strip()
                    coords_part = parts[1].strip()
                    
                    # Extract index (1-based in YOLO output)
                    try:
                        index = int(index_part.split()[1]) - 1  # Convert to 0-based index
                        if 0 <= index < len(keypoint_names):
                            keypoint_name = keypoint_names[index]
                            
                            # Extract coordinates
                            coords = coords_part.split(',')
                            x = int(coords[0].split('=')[1])
                            y = int(coords[1].split('=')[1])
                            
                            keypoint_dict[keypoint_name] = [x, y]
                    except (ValueError, IndexError):
                        # Skip malformed lines
                        pass
    except Exception as e:
        print(f"Error parsing keypoints: {str(e)}")
        # Return empty keypoints rather than failing
    # print(keypoint_dict)
    return {"keypoints": keypoint_dict}

def add_debugging_info(result, image_shape, skin_regions=None):
    """
    Add debugging information to the result dictionary
    
    Args:
        result: Result dictionary
        image_shape: Shape of the processed image
        skin_regions: List of skin regions
        
    Returns:
        Updated result dictionary
    """
    if "error" in result:
        result["debug_info"] = {
            "image_shape": image_shape,
            "skin_regions": skin_regions if skin_regions else "None detected"
        }
    return result

if __name__ == "__main__":
    # Read input from stdin (JSON with base64 image and optional keypoints)
    try:
        input_data = sys.stdin.read()
        parsed = json.loads(input_data)
        base64_image = parsed['image']
        
        # Process the base64 image
        try:
            # Convert base64 to image array
            image_array = process_base64_image(base64_image)
            
            # Get keypoints if provided
            keypoints = None
            if 'keypoints_text' in parsed and parsed['keypoints_text']:
                keypoints = parse_yolo_keypoints(parsed['keypoints_text'])
            else:
                # Use default keypoints based on the example
                keypoints = {"keypoints": {}}
            
            # Detect skin tone
            skin_tone_info = detect_skin_tone_from_image(image_array, keypoints)
            
            # Add debugging info if error
            if "error" in skin_tone_info:
                skin_tone_info = add_debugging_info(skin_tone_info, image_array.shape)
            
            # Output the result as JSON
            print(json.dumps(skin_tone_info))
            
        except Exception as e:
            error_result = {"error": f"Failed to process image: {str(e)}"}
            print(json.dumps(error_result))
            
    except Exception as e:
        error_result = {"error": f"Failed to parse input: {str(e)}"}
        print(json.dumps(error_result))