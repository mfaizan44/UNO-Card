# UNO-Card
Readme File

Please install the prequisite.
install python
install jupyter notebook
install pip
install cv2, numpy, tensorflow etc.

In my case I am using Python3 for Jupyter notebook
Go to command line and write 'jupyter notebook'

Initialise Jupter notebook
and create a new python 3 program





#import all the directories and define the terms.
import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the path to the directory containing the hand gesture dataset
data_dir = r'C:\Users\HP\Pictures\Camera Roll\UNOCARD'

# Set the dimensions of the input images
img_width, img_height = 224, 224

# Set the batch size for training
batch_size = 16

# Create a data generator to augment the training data
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Create a data generator for the validation data
val_datagen = ImageDataGenerator(rescale=1./255)

# Create the training dataset generator
train_generator = train_datagen.flow_from_directory(
    os.path.join(r'C:\Users\HP\AppData\Local\Programs\Python\Python311\Scripts', 'train'),
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Create the validation dataset generator
val_generator = val_datagen.flow_from_directory(
    os.path.join(data_dir, 'val'),
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

Once this code is run the system detects all the samples need to be trained.
in my case 
'Found 3191 images belonging to 53 classes.
Found 3222 images belonging to 53 classes.'





Now we have to create and train our model.
# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(53, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.n // batch_size,
                    epochs=50,
                    validation_data=val_generator,
                    validation_steps=val_generator.n // batch_size)

# Save the model
model.save('unocardmodel.h5')




Once the unocardmodel.h5 is saved we can start to compare from the real time web camera
import cv2
import numpy as np
from keras.models import load_model

# Define the dimensions of the input image
img_width, img_height = 224, 224

# Load the trained model
model = load_model('unocardmodel.h5')

# Convert the image to the format expected by the model
def preprocess_image(img):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert the grayscale frame to BGR
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # Resize the BGR frame to the dimensions expected by the model
    img = cv2.resize(img, (img_width, img_height))
    # Scale the pixel values to the range [0, 1]
    img = img / 255.0
    # Reshape the image to have a single channel
    img = np.reshape(img, (img_width, img_height, 3))
    # Return the preprocessed image
    return img

# Define a function to predict the label of the input image
def predict_label():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Set the threshold for the predicted probability
    threshold = 0.5

    # Loop over the frames from the webcam
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Preprocess the frame
        img = preprocess_image(frame)

        # Make a prediction using the model
        pred = model.predict(np.array([img]))

        # Get the index of the class with the highest predicted probability
        class_idx = np.argmax(pred)

        # Define the class labels
        labels = ['blue0', 'blue1', 'blue2','blue3','blue4','blue5','blue6','blue7','blue8','blue9','bluedraw2','bluereverse','blueskip','draw4','green0','green1','green2','green3','green4','green5','green6','green7','green8','green9','greendraw2','greenreverse','greenskip','red0','red1','red2','red3','red4','red5','red6','red7','red8','red9','reddraw2','redreverse','redskip','wild','yellow0','yellow2','yellow3','yellow4','yellow5','yellow6','yellow7','yellow8','yellow9','yellowdraw2','yellowreverse','yellowskip']

        # Get the predicted label and probability
        label = labels[class_idx]
        prob = pred[0][class_idx]

        # Check if the predicted probability is above the threshold
        if prob >= threshold:
            # Draw a bounding box around the detected card
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)

            # Draw the predicted label and probability on the frame
            text = f"{label} ({prob:.2f})"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            # If no card is detected, display a message on the frame
            cv2.putText(frame, "No unocard detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow('frame', frame)

        # Check if the user pressed the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Call the predict_label function
predict_label()


frame window of the front facing camera will open and start capturing images and comparing it to the dataset.
press q to exit. 







Please note my github repository link is : https://github.com/mfaizan44/UNO-Card/tree/main
and my demo video link is : https://drive.google.com/file/d/13lhxlsiWEkhWQxtfyxkqGdenSS8Wvp-T/view?usp=sharing
