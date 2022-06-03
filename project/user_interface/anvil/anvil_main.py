import anvil.server
import anvil.media
import io
from PIL import Image
from TT_SVD_1 import tensortrain, tt_reconstruction, border
import numpy as np


anvil.server.connect('DH4R5MH3DTKYNYBPF63XKZLV-UHSXWMNTP7IBV2RQ')

@anvil.server.callable
def process_image(image):
    image_object = Image.open(io.BytesIO(image.get_bytes()))
    image = np.asarray(image_object)
    return image

@anvil.server.callable
def image_tensor_train(img, epsilon=0.1):
    g, d, r, n = tensortrain(img, epsilon)
    reconstructed = tt_reconstruction(g, d, r, n)
    new_image = border(np.array(reconstructed), 0, 255, p=True)
    final_image = Image.fromarray(new_image.astype(np.uint8), 'RGB')
    bs = io.BytesIO()
    name = 'final_image'
    final_image.save(bs, format="JPEG")
    final = anvil.BlobMedia("image/jpeg", bs.getvalue(), name=name)
    return g, d, r, n, final


anvil.server.wait_forever()
