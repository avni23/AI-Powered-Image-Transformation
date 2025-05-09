import os
import torch
from torchvision import transforms, models
from PIL import Image
from nst_utils import load_image, save_image, normalize_batch  # You already have these

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def neural_style_transfer(config):
    content_path = os.path.join(config["content_images_dir"], config["content_img_name"])
    style_path = os.path.join(config["style_images_dir"], config["style_img_name"])
    output_dir = config["output_img_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess images
    content_image = load_image(content_path, size=config["height"])
    style_image = load_image(style_path, size=config["height"])
    
    transform = transforms.ToTensor()
    content_tensor = transform(content_image).unsqueeze(0).to(device)
    style_tensor = transform(style_image).unsqueeze(0).to(device)

    content_tensor = normalize_batch(content_tensor)
    style_tensor = normalize_batch(style_tensor)

    # Load pre-trained VGG
    vgg = models.vgg19(pretrained=True).features.to(device).eval()

    # Feature extraction
    layers = {
        '0': 'conv1_1', '5': 'conv2_1',
        '10': 'conv3_1', '19': 'conv4_1',
        '21': 'conv4_2', '28': 'conv5_1'
    }

    content_features = {}
    style_features = {}
    x = content_tensor.clone()
    y = style_tensor.clone()

    for name, layer in vgg._modules.items():
        x = layer(x)
        y = layer(y)
        if name in layers:
            if layers[name] == 'conv4_2':
                content_features[layers[name]] = x
            style_features[layers[name]] = y

    # Gram matrices
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # Generated image
    generated = content_tensor.clone().requires_grad_(True)

    optimizer = torch.optim.LBFGS([generated])

    run = [0]
    while run[0] <= 300:
        def closure():
            optimizer.zero_grad()
            gen_features = {}
            x = generated

            for name, layer in vgg._modules.items():
                x = layer(x)
                if name in layers:
                    gen_features[layers[name]] = x

            content_loss = torch.nn.functional.mse_loss(
                gen_features['conv4_2'], content_features['conv4_2'])

            style_loss = 0
            for layer in style_grams:
                gen_gram = gram_matrix(gen_features[layer])
                style_gram = style_grams[layer]
                style_loss += torch.nn.functional.mse_loss(gen_gram, style_gram)

            total_loss = config["content_weight"] * content_loss + \
                         config["style_weight"] * style_loss
            total_loss.backward()
            run[0] += 1
            return total_loss

        optimizer.step(closure)

    # Save result
    output_image = generated.cpu().clone().squeeze()
    output_dir = os.path.join(output_dir, f'combined_{config["content_img_name"].split(".")[0]}_{config["style_img_name"].split(".")[0]}')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "stylized_result.jpg")
    save_image(output_path, output_image)
    return output_path
