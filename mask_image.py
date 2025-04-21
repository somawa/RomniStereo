import cv2
import numpy as np

# Define the size of the mask region and maximum window size
MASK_SIZE = 200  # This defines the diameter of the circular mask
MAX_WINDOW_WIDTH = 800
MAX_WINDOW_HEIGHT = 600

# Function to resize image while maintaining aspect ratio
def resize_image(image, max_width, max_height):
    h, w = image.shape[:2]
    scaling_factor = min(max_width / w, max_height / h)
    new_size = (int(w * scaling_factor), int(h * scaling_factor))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_image, scaling_factor

# Callback function to handle mouse click events
def select_pixel(event, x, y, flags, param):
    global mask, image_display, scaling_factor, original_image
    if event == cv2.EVENT_LBUTTONDOWN:
        # Scale the click coordinates back to the original image size
        orig_x = int(x / scaling_factor)
        orig_y = int(y / scaling_factor)

        # Draw a circle on the mask
        cv2.circle(mask, (orig_x, orig_y), MASK_SIZE // 2, 255, -1)  # Draw a filled circle with radius MASK_SIZE//2

        # Draw a red circle on the display image to mark the selected region
        resized_x = int(orig_x * scaling_factor)
        resized_y = int(orig_y * scaling_factor)
        resized_radius = int((MASK_SIZE // 2) * scaling_factor)
        cv2.circle(image_display, (resized_x, resized_y), resized_radius, (0, 0, 255), -1)  # Red circle on the display image

# Load the image
original_image = cv2.imread('input_image.jpg')
if original_image is None:
    print("Error: Could not open or find the image.")
    exit()

# Resize the image to fit within the maximum window size and get the scaling factor
image, scaling_factor = resize_image(original_image, MAX_WINDOW_WIDTH, MAX_WINDOW_HEIGHT)

# Create a copy of the resized image to display
image_display = image.copy()

# Initialize a mask with the same dimensions as the original image, filled with zeros (black)
mask = np.zeros(original_image.shape[:2], dtype=np.uint8)

# Create a window and set the mouse callback function
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', select_pixel)

while True:
    # Display the resized image
    cv2.imshow('Image', image_display)

    # Wait for the 'q' key to be pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Apply the mask to the original image
masked_image = cv2.bitwise_and(original_image, original_image, mask=cv2.bitwise_not(mask))

# Save the masked image
# cv2.imwrite('masked_image.jpg', masked_image)

# Save the mask itself
mask_image = cv2.bitwise_and(original_image, original_image, mask=mask)
_, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
binary_mask = 255 - binary_mask
binary_mask = binary_mask.astype(np.uint8)
cv2.imwrite('mask.png', binary_mask)

# Clean up and close windows
cv2.destroyAllWindows()

# Display final masked image
cv2.imshow('Masked Image', masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
