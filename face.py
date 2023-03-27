import cv2
import os

# Créer le dossier où les images de visages seront sauvegardées
if not os.path.exists('faces'):
    os.makedirs('faces')

# Charger le classificateur de cascade pour la détection des visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Parcourir les fichiers dans le répertoire donné
for file_name in os.listdir('img'):

    if not file_name.endswith('.jpg'):  # Ignorer les fichiers qui ne sont pas des images JPEG
        continue
    file_path = os.path.join('img', file_name)
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Si aucun visage n'est détecté, ignorer l'image
    if len(faces) == 0:
        continue
    
    # Parcourir les visages détectés et les sauvegarder dans le dossier "faces"
    for (x, y, w, h) in faces:
        face_image = image[y:y+h, x:x+w]
        face_file_name = os.path.splitext(file_name)[0] + '_face.jpg'
        face_file_path = os.path.join('faces', face_file_name)
        cv2.imwrite(face_file_path, face_image)
