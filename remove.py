import cv2
import os

def process_image(input_path, output_path, action="blur"):
    """
    Detect faces in an image and either blur or remove them.
    
    Parameters:
        input_path (str): Path to the input image.
        output_path (str): Path to save the processed image.
        action (str): Action to perform on detected faces ('blur' or 'remove').
    """
    # Load the image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Unable to load the image at {input_path}.")
        return

    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"Detected {len(faces)} face(s) in {input_path}.")

    # Process each detected face
    for (x, y, w, h) in faces:
        if action == "blur":
            # Blur the face
            face_region = image[y:y+h+20, x:x+w+20]
            blurred_face = cv2.GaussianBlur(face_region, (51, 51), 30)
            image[y:y+h+20, x:x+w+20] = blurred_face
        elif action == "remove":
            # Remove the face by filling it with a solid color (black)
            image[y:y+h, x:x+w] = (0, 0, 0)

    # Save the processed image
    cv2.imwrite(output_path, image)
    print(f"Processed image saved to: {output_path}")


# Configure paths
input_folder = "/Users/admin/Downloads/suit"
output_folder = "/Users/admin/Downloads/suit1"


# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Action to perform: "blur" or "remove"
action = "blur"  # Change to "remove" if you want to remove faces

# Process all images in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', 'webp', 'avif')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        process_image(input_path, output_path, action)

print("All images processed.")

 
