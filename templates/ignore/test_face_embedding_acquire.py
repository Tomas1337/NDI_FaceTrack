
import sys
sys.path.insert(0, 'face_tracking/')
from face_recognition import FaceRecognizer, SAVED_EMBEDDINGS_PATH, SAVED_NAMES_PATH
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from pathlib import Path
import random
import string
import os
import cv2 as cv
"""
Tests for face_recognition.py 
"""

face_recognizer = FaceRecognizer()

def test_get_face_embedding():

    mtcnn = MTCNN(margin = 14, factor = 0.6, keep_all= True,post_process=True, select_largest=False,device= 'cpu')
    
    image_path = "test/test_content/test_jpg.jpg"
    image = Image.open(image_path)
    image = np.array(image)

    faces = mtcnn.forward(image)
    face_embedding = face_recognizer.get_face_embedding(faces)

    return face_embedding

def test_save_embedding():
    
    face_embedding = test_get_face_embedding()

    #Generate random name
    name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    face_recognizer.save_embedding(face_embedding, name)

    emb_path = Path(SAVED_EMBEDDINGS_PATH)
    with emb_path.open('rb') as f:
        embeddings = np.load(f)
        last_embedding = embeddings[-1]
        assert (last_embedding == face_embedding.numpy()).all()

    #Check if name is saved inside
    name_path = Path(SAVED_NAMES_PATH)
    with name_path.open('rb') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
        assert last_line.decode("utf-8") == name

def test_clear_saved_names_embeddings():
    face_recognizer.clear_saved_names_embeddings()

    assert (os.stat(SAVED_EMBEDDINGS_PATH).st_size == 0)
    assert (os.stat(SAVED_NAMES_PATH).st_size == 0)

def test_find_embedding_match():
    #Setup
    face_recognizer.clear_saved_names_embeddings()
    
    #get an embedding from the test_jpg.jpg and saves it
    face_embedding = test_get_face_embedding()      
    face_recognizer.save_embedding(face_embedding)
    name, score = face_recognizer.find_embedding_match(face_embedding)
    print (name, score)


def test_find_embedding_matches():
    face_recognizer.clear_saved_names_embeddings()
    mtcnn = MTCNN(image_size = 160, margin = 14, factor = 0.6, keep_all= True,post_process=True, select_largest=False,device= 'cpu')
    threshold = 0.8
    name_counter = 1
    #Generator object through files
    pathlist = Path("test/test_content").glob('**/*.jpg')

    #Open Each Test Picture
    for file in pathlist:
        identity_name = file.stem 
        frame = Image.open(file)
    
        #Run through MTCNN network
        faces, prob = mtcnn.forward(frame, return_prob= True)

        #Iterate through faces
        for idx, face in enumerate(faces):
            if prob[idx] > threshold:
                embedding = face_recognizer.get_face_embedding(face.permute(1,2,0).numpy())
                name, score = face_recognizer.find_embedding_match(embedding)

                if name is None:
                    face_recognizer.save_embedding(embedding, name = identity_name + str(name_counter))
                    print(f'Saving embedding with name {identity_name + str(name_counter)}')
                    name_counter += 1

                else:
                    print(f'Match found for {name} in image {}')


if __name__ == "__main__":
    # test_get_face_embedding()
    # test_save_embedding(  )
    # test_clear_saved_names_embeddings()
    #test_find_embedding_match()
    test_find_embedding_matches()
    


