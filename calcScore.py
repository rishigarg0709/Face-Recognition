from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
import numpy as np
from keras import backend as K

def get_model_scores(faces):
    samples = np.array(faces, 'float32')

    samples = preprocess_input(samples, version=2)

    model = VGGFace(model='resnet50',
      include_top=False,
      pooling='avg')

    return model.predict(samples.reshape(1,224,224,3))

