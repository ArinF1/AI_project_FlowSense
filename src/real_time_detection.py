import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from smoothing import LabelSmoother

# config
MODEL_PATH = Path("src/models/mobilenetv2_best.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
FONT = cv2.FONT_HERSHEY_SIMPLEX

def load_trained_model(path, device):

    #Load out trained model

    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    
    #load the checkpoint dictionary
    checkpoint = torch.load(path, map_location=device)
    class_names = checkpoint['classes']
    num_classes = len(class_names)
    
    #Build the model architecture no pretrained weights
    model = models.mobilenet_v2(weights=None)
    

    # final layer replacement
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
    # load the weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval() # Set to eval
    
    print(f"Model is loaded, detecting classes: {class_names}")
    return model, class_names


# function to get preprocessing transforms
def get_transforms():
    # Standard ImageNet normalization used in training
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        norm
    ])

# function to predict on a single frame
def predict_frame(model, frame, transform, class_names, device):

    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    pil_img = Image.fromarray(rgb_frame) # Convert to PIL Image
    
    # Apply transforms
    input_tensor = transform(pil_img).unsqueeze(0) # Add batch dimension
    input_tensor = input_tensor.to(device) # to device
    
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    # label and confidence score    
    label = class_names[predicted_idx.item()]
    conf_score = confidence.item()
    return label, conf_score


def main():
    #load Model
    try:
        model, class_names = load_trained_model(MODEL_PATH, DEVICE)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Webcam setup
    cap = cv2.VideoCapture(0) # Try index 1 if 0 doesn't work
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Preprocessing
    transform = get_transforms()

    # Initializs smoother
    smoother = LabelSmoother(buffer_size=12)
    
    print("FlowSense detection started")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # CALCULATE CENTER CROP
        h, w, _ = frame.shape
        size = h
        start_x = (w - size) // 2
        cropped_frame = frame[:, start_x:start_x+size] # 1:1 Aspect Ratio

        # run prediction
        raw_label, confidence = predict_frame(model, cropped_frame, transform, class_names, DEVICE)

        label = smoother.update(raw_label)

        # Visualize
        # confident = green, yellow = less confident
        color = (0, 255, 0) if confidence > 0.6 else (0, 255, 255)
        
        # display text
        text = f"State: {label.upper()} ({confidence*100:.1f}%)"
        cv2.putText(frame, text, (10, 50), FONT, 1, color, 2, cv2.LINE_AA)
        
        # rectangle border
        cv2.rectangle(frame, (5, 5), (frame.shape[1]-5, frame.shape[0]-5), color, 2)

        cv2.imshow('FlowSense', frame)

        #exit with q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()