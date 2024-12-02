import streamlit as st
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = 0.485, 0.456, 0.406
IMAGENET_STD = 0.229, 0.224, 0.225


def classify_transforms(size=224):
    # Applies a series of image transformations to prepare an image for classification.

    #     The transformations include:
    #     - Converting the image to a PyTorch tensor
    #     - Resizing the image to the specified size
    #     - Centrally cropping the image to the specified size
    #     - Normalizing the image using the ImageNet mean and standard deviation

    #     This function is typically used as part of a PyTorch data pipeline for image classification tasks.
    return T.Compose(
        [
            T.ToTensor(),
            T.Resize(size=size),
            T.CenterCrop(size),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def load_yolo_model():
    yolo_model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")
    print(type(yolo_model))
    return yolo_model


model_YOLO = load_yolo_model()


def detect_faces(image):
    column_x, column_y = 10, 30
    rect_color = (0, 255, 0)
    # Load the pre-trained Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Draw rectangles around each face
    for x, y, w, h in faces:
        face = gray[y : y + h, x : x + w]
        # face = cv2.resize(face, yolo_model_input_size)
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        tranformations = classify_transforms()
        convert_tensor = tranformations(face)
        convert_tensor = convert_tensor.unsqueeze(0)
        convert_tensor = convert_tensor.to(device)
        print(convert_tensor.shape)
        results = model_YOLO(convert_tensor)
        pred = F.softmax(results, dim=1)
        _, max_index = torch.max(pred, 1)
        prediction = model_YOLO.names[max_index.item()]
        cv2.putText(
            image,
            prediction,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        for i, prob in enumerate(pred):
            top5i = prob.argsort(0, descending=True)[:5].tolist()
            for j in top5i:
                text = f"{prob[j]:.2f} {model_YOLO.names[j]}"
                cv2.putText(image, text, (column_x, column_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rect_color, 2)  # type: ignore
                column_y += 25  # Adjust this value based on the desired vertical spacing between lines
            column_x += 150  # Adjust this value based on the desired horizontal spacing between columns
            column_y = 100

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image


def main():
    st.title("Face Detector")
    st.sidebar.title("Settings")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Image", "Camera"])

    if app_mode == "Image":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Image", type=["jpg", "png", "jpeg"]
        )

        if uploaded_file is not None:
            image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
            st.image(
                image, channels="BGR", caption="Original Image", use_column_width=True
            )

            if st.sidebar.button("Detect Faces"):
                result_image = detect_faces(image)
                st.image(
                    result_image,
                    channels="BGR",
                    caption="Image with Detected Faces",
                    use_column_width=True,
                )

    elif app_mode == "Camera":
        st.sidebar.markdown("Press 'Start' to begin capturing video from your webcam.")

        run = st.sidebar.checkbox("Start")
        FRAME_WINDOW = st.image([])

        # Access the webcam
        video_capture = cv2.VideoCapture(0)  # Change camera index to 0

        while run:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            if not ret:
                st.error(
                    "Failed to capture frame from webcam. Please check your webcam connection."
                )
                break

            # Detect faces
            frame_with_faces = detect_faces(frame)

            # Display the resulting frame
            FRAME_WINDOW.image(frame_with_faces, channels="BGR", caption="Video Feed")

        # Release the webcam feed
        video_capture.release()

    else:
        st.warning("No option selected")


if __name__ == "__main__":
    main()
