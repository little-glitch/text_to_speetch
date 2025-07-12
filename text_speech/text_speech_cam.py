#import packages and classification model
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import cv2
import pyttsx3
import RPi.GPIO as GPIO
from time import sleep

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

engine = pyttsx3.init()
voice = engine.getProperty('voices')
engine.setProperty('voice', voice[3].id) #changing voice to index 1 for female voice
engine.setProperty("rate", 125)

# define a video capture object
vid = cv2.VideoCapture(0)

#load TFLite model
interpreter = tf.lite.Interpreter('/home/pi/project/text_speech/model.tflite')
print(interpreter.get_signature_list())
classify_lite = interpreter.get_signature_runner('serving_default')

while(True):
    # Capture the video frame, by frame
    ret, frame = vid.read()
    # Display the resulting frame
    cv2.imshow('frame', frame)

    #preprocess the image
    img_pre=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pre=cv2.resize(img_pre, (224,224))
    img_pre = np.expand_dims(img_pre, axis=0)
    img_pre = preprocess_input(img_pre)
    preds = classify_lite(input_1=img_pre)['predictions']

    if GPIO.input(7) == GPIO.HIGH:
        print('Predicted:', decode_predictions(preds, top=3)[0][0][1].replace("_", " "))
        text = decode_predictions(preds, top=3)[0][0][1].replace("_", " ")
        engine.say(text)
        engine.runAndWait()
        

    #predict the object using the model
        """
    preds = classify_lite(input_1=img_pre)['predictions']
    print('Predicted:', decode_predictions(preds, top=3)[0][0][1].replace("_", " "))
    text = decode_predictions(preds, top=3)[0][0][1].replace("_", " ")
    engine.say(text)
    engine.runAndWait()
    """
    # the 'q' button is set as the quiting button
    if cv2.waitKey(1) & 0xFF == ord('q'):
    	break

vid.release()
cv2.destroyAllWindows()
