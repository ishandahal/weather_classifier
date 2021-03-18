from torchvision import transforms
from PIL import Image

def image_transformer(image_path):
    """Takes a Jpeg or Png image, resizes
    to 224*224 and returns image tensor of
    channel*height*width dimension
    """
    image = Image.open(image_path)
    image = transforms.Resize([224, 224])(image)
    image = transforms.ToTensor()(image)
    if image.size(0) != 3:
        image = image[0:3, :, :]
    
    return image



# print(image_tensor.size())
# image = transforms.Resize([224,224])(image)
# image = transforms.ToTensor()(image)
# if image.size(0) != 3:
#     image = image[0:3, :, :]
# print(image.size())