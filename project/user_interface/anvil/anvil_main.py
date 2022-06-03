import anvil.server
import anvil.media
import io
from PIL import Image
from TT_SVD_1 import tensortrain, tt_reconstruction, border
import numpy as np
import pandas as pd
import anvil.tables as tables
from anvil.tables import app_tables

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


@anvil.server.callable
def import_csv_data(file):
    with anvil.media.TempFile(file) as f:
        df = pd.read_csv(f)
        df = df.dropna()
        for d in df.to_dict(orient="records"):
            # d is now a dict of {columnname -> value} for this row
            # We use Python's **kwargs syntax to pass the whole dict as
            # keyword arguments
            app_tables.table_0.add_row(**d)

@anvil.server.callable
def clear_table():
    app_tables.table_0.delete_all_rows()



anvil.server.wait_forever()
