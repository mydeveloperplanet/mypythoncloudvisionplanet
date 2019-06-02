import io
from google.cloud import vision
from PIL import Image, ImageDraw


def detect_labels(path):

    print('#################')
    print('# Detect Labels #')
    print('#################')

    """Detects labels in the file."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations
    print('Labels:')

    for label in labels:
        print(label.description)


def detect_text(path):
    print('###############')
    print('# Detect Text #')
    print('###############')

    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')

    for text in texts:
        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

        print('bounds: {}'.format(','.join(vertices)))


def detect_landmarks(path):

    print('###################')
    print('# Detect Landmark #')
    print('###################')

    """Detects landmarks in the file."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.landmark_detection(image=image)
    landmarks = response.landmark_annotations
    print('Landmarks:')

    for landmark in landmarks:
        print(landmark.description)
        for location in landmark.locations:
            lat_lng = location.lat_lng
            print('Latitude {}'.format(lat_lng.latitude))
            print('Longitude {}'.format(lat_lng.longitude))


def detect_faces():

    print('################')
    print('# Detect Faces #')
    print('################')

    input_filename = "faces.jpeg"
    output_filename = "output.jpeg"
    with open(input_filename, 'rb') as image:
        faces = detect_face(image)
        print('Found {} face{}'.format(
            len(faces), '' if len(faces) == 1 else 's'))

        print('Writing to file {}'.format(output_filename))
        # Reset the file pointer, so we can read the file again
        image.seek(0)
        highlight_faces(image, faces, output_filename)


def detect_face(face_file, max_results=4):
    client = vision.ImageAnnotatorClient()

    content = face_file.read()
    image = vision.types.Image(content=content)

    return client.face_detection(image=image, max_results=max_results).face_annotations


def highlight_faces(image, faces, output_filename):
    im = Image.open(image)
    draw = ImageDraw.Draw(im)
    # Specify the font-family and the font-size
    for face in faces:
        box = [(vertex.x, vertex.y)
               for vertex in face.bounding_poly.vertices]
        draw.line(box + [box[0]], width=5, fill='#00ff00')
        # Place the confidence value/score of the detected faces above the
        # detection box in the output image
        draw.text(((face.bounding_poly.vertices)[0].x,
                   (face.bounding_poly.vertices)[0].y - 30),
                  str(format(face.detection_confidence, '.3f')) + '%',
                  fill='#FF0000')
    im.save(output_filename)


def main():
    detect_labels("cat.jpg")
    detect_text("text.jpeg")
    detect_landmarks("landmark.jpeg")
    detect_faces()


if __name__== "__main__":
    main()