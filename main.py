import cv2
from deepface import DeepFace
import argparse
import os

def analyze_and_annotate(image_path, output_path="annotated_image.jpg"):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image")
        return

    # Analyze image using DeepFace
    results = DeepFace.analyze(img_path=image_path, actions=['gender', 'race'], enforce_detection=False)

    # Iterate over results and annotate the image
    for res in results:
        gender = res['dominant_gender']
        ethnicity = res['dominant_race']
        region = res['region']  # Bounding box coordinates

        # Draw bounding box
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Add label text
        label = f"{gender}, {ethnicity}"
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the annotated image
    cv2.imwrite(output_path, img)
    print(f"Annotated image saved as {output_path}")

    # Display the image
    cv2.imshow("Annotated Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gender and Ethnicity Classifier with Annotation")
    parser.add_argument("--image", type=str, help="Path to the image")
    parser.add_argument("--video", type=str, help="Path to the video")
    parser.add_argument("--output", type=str, default="annotated_image.jpg", help="Output path for annotated image")

    args = parser.parse_args()

    if args.image:
        analyze_and_annotate(args.image, args.output)
    elif args.video:
        print("Video processing not implemented yet.")
    else:
        print("Please provide an image or video path")
