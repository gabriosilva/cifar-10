from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from d2l import torch as d2l
import io

app = Flask(__name__)

# Define the transform for the incoming images
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

# Define the label map
label_map = {
  0: 'airplane',
  1: 'automobile',
  2: 'bird',
  3: 'cat',
  4: 'deer',
  5: 'dog',
  6: 'frog',
  7: 'horse',
  8: 'ship',
  9: 'truck'
}


def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net

def save_model(model, path):
    """Save the model to the specified path."""
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    """Load the model from the specified path."""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = get_net()
model_path = 'model/cifar10_resnet18.pth'  # Replace with your model path
load_model(net, model_path, device)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    img = transform(img).unsqueeze(0).to(device)
  
    net.eval()
    with torch.no_grad():
        output = net(img)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
        label = predicted.item()
  
    label_name = label_map[label]
    label_probabilities = {label_map[i]: prob for i, prob in enumerate(probabilities.tolist()[0])}

    return jsonify({'label': label_name, 'probabilities': label_probabilities})

if __name__ == '__main__':
    app.run(debug=True)

