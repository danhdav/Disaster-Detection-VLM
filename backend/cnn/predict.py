import io
import torch
from torchvision import transforms
from PIL import Image

from .model import DisasterCNN, DISASTER_CLASSES


_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(weights_path: str) -> DisasterCNN:
    """Load a trained DisasterCNN from a .pt weights file."""
    model = DisasterCNN(num_classes=len(DISASTER_CLASSES), pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


def predict_image(model: DisasterCNN, image_bytes: bytes) -> dict:
    """
    Run inference on raw image bytes.
    Returns a dict with the predicted class label and confidence scores.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _TRANSFORM(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze().tolist()

    predicted_idx = int(torch.argmax(logits, dim=1).item())
    return {
        "prediction": DISASTER_CLASSES[predicted_idx],
        "confidence": round(probabilities[predicted_idx], 4),
        "scores": {cls: round(prob, 4) for cls, prob in zip(DISASTER_CLASSES, probabilities)},
    }
