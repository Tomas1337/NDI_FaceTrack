from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
#import pandas as pd
import os
from pathlib import Path
from PySide2.QtCore import QObject

SAVED_EMBEDDINGS_PATH = "models/identity_embeddings.npz"
SAVED_NAMES_PATH = "models/identity_names.txt"
FACE_SIZE = 160

class FaceRecognizer(QObject):
    def __init__(self, parent= None):
        super().__init__(parent)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        print('Running on device: {}'.format(self.device))  

    def get_face_embedding(self, image: np.ndarray):
        if image.shape == (FACE_SIZE, FACE_SIZE, 3):
            image = image.transpose(2, 0, 1)
            image = image[None, :, :, :]
        
        try:
            image = image.to(self.device)
            face_embedding = self.resnet(image).detach().cpu()

        except (TypeError, AttributeError) as e:
            image = torch.tensor(image)
            image = image.to(self.device)
            face_embedding = self.resnet(image).detach().cpu()

        return face_embedding

    def save_embedding(self, embedding, name = None):
        if name is None:
            name = "Mr Kenobi"

        #Save Embedding to file
        emb_path = Path(SAVED_EMBEDDINGS_PATH)

        if os.stat(emb_path).st_size != 0:
            #Read if there are any present embedding files inside
            emb_dict = self._npz_to_dict(emb_path)
            emb_dict[name] = (embedding)
            np.savez(emb_path, **emb_dict)
        else:
            np.savez(emb_path, **{name: embedding.numpy()})

    def _npz_to_dict(self, npz_file_path):
            #Helper function to unpack a npz file to an appendable dictionary
            emb_dict = {}
            emb_npz = np.load(npz_file_path)

            for emb in emb_npz.files:
                emb_dict[emb]= emb_npz[emb]

            #print(f'Embedding Dictionary: {emb_dict}')
            return emb_dict

    def find_embedding_match(self, embedding, match_threshold = 0.6):
        # Read face embeddings file

        emb_path = Path(SAVED_EMBEDDINGS_PATH)
        if os.stat(emb_path).st_size != 0:
            embeddings_npz = np.load(emb_path, allow_pickle = True)
            print(f'There are {len(embeddings_npz.files)} embeddings stored')
            dists = [np.linalg.norm(embedding - embeddings_npz[e2]) for e2 in embeddings_npz]
            identity_index = dists.index(min(dists))
            matched_score = dists[identity_index]

            if matched_score > match_threshold:
                matched_name = None
                matched_score = None

            else:   
                matched_name =  list(embeddings_npz.keys())[identity_index]
                
        else:
            matched_name = None
            matched_score = None
                    
        return matched_name, matched_score

    def clear_saved_names_embeddings(self):
        def deleteContent(pfile):
            pfile.seek(0)
            pfile.truncate()

        names_path = Path(SAVED_NAMES_PATH)
        with names_path.open('a') as name_file:
            deleteContent(name_file)

        emb_path = Path(SAVED_EMBEDDINGS_PATH)
        with emb_path.open('ab') as emb_file:
            deleteContent(emb_file)

    def modify_name(self, old_name, new_name):

        emb_path = Path(SAVED_EMBEDDINGS_PATH)

        #Read if there are any present embedding files inside
        if os.stat(emb_path).st_size != 0:
            
            emb_dict = self._npz_to_dict(emb_path)
            emb_dict[new_name] = emb_dict.pop(old_name)
            np.savez(emb_path, **emb_dict)
            return True

        else:
            return False



