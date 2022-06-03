import anvil.server
import anvil.media
import io
from PIL import Image


# dog: "C:\Users\tommo\Downloads\dog.jpg.jpeg"

anvil.server.connect('DH4R5MH3DTKYNYBPF63XKZLV-UHSXWMNTP7IBV2RQ')

@anvil.server.callable
def get_image_path(file):
    img = file
    return img

# @anvil.server.callable
# def image_tensor_train(img, epsilon):
#     g, d, r, n = tensortrain(img, epsilon)
#     reconstructed = tt_reconstruction(g, d, r, n)
#     final_image = border(np.array(reconstructed), 0, 255, p=True)
#     return g, d, r, n, final_image


anvil.server.wait_forever()
