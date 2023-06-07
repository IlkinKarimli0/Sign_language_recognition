from Data_Generator import DataLoader
import numpy as np
import tensorflow as tf
from keras import backend as K
import pickle 
from keras_i3d import I3D
import cv2
from PIL import ImageFont, ImageDraw, Image  


print(K.backend())


model = I3D().model(classes = 261)

model.load_weights(r'/home/nigar.alishzada/SLR/keras-kinetics-i3d/weights_all/weights-08-1.54.h5')


#Video path to load
video_path = r'/home/nigar.alishzada/SLR/keras-kinetics-i3d/2022-06-01 15-17-31.mp4'

#Preprocess the video as it done in the training 
video = DataLoader.frames_from_video_file(video_path,90,frame_step = 1)


#Number of the input frame is 20 for model 
clip_size = 20

# You can change stride based your intuition 
stride = 20
num_clips = (90 - clip_size) // stride + 1

#Declare the np array in memory
clips = np.zeros((num_clips, clip_size, 224, 224, 3))
start_frames = []


# cropping the clips from video and add them into clips array 
for i in range(num_clips):
    clips[i] = video[i*stride:i*stride+clip_size]
    #get start frame of the each 
    start_frames.append(i*stride)

#Load index to label dict to get label string from softmax output
with open('index_to_label.pkl','rb') as fb:
    index_to_label = pickle.load(fb)



# Set path to font file
font_path = r"/home/nigar.alishzada/SLR/keras-kinetics-i3d/FreeSans.ttf"

# Initialize empty lists
predictions = []
pred_start_frames = []

# Loop over start_frames
for i in start_frames:

    # Select video clip
    clip = clips[int(i/stride)-1]

    # Expand clip along first axis
    clip = np.expand_dims(clip, axis=0)

    # Predict label for clip
    prediction = model.predict(clip)
    index = np.argmax(prediction)
    confidence = prediction[0][index]

    # Append predicted label and starting frame to lists
    if confidence > 0:
        predictions.append(index_to_label[index])
        pred_start_frames.append(i)
    else:
        pass

# Delete start_frames list
del start_frames

# create a window to display the video
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', 800, 600) # set window size as desired

# Define empty prediction variable
prediction = ""

# Loop over the video frames and enumerate them
for i, frame in enumerate(video):

    # If the current frame index is in the list of prediction start frames, get the corresponding prediction
    if i in pred_start_frames:
        prediction = predictions[int(i/stride)]

    # Convert the frame from NumPy array to PIL Image
    pil_im = Image.fromarray((frame * 255).astype(np.uint8))

    # Create a PIL ImageDraw object to draw text on the PIL Image
    draw = ImageDraw.Draw(pil_im) 

    # Load the font and set its size to 15
    font = ImageFont.truetype(r"C:\Users\ii.karimli\Desktop\sign language\final version\FreeSans.ttf", 15)  
    
    # Draw the text on the PIL Image at position (25, 25) with the loaded font
    draw.text((25, 25), prediction, font=font)  
   
    # Convert the PIL Image back to NumPy array and then to BGR format
    cv2_im_processed = cv2.cvtColor(np.array(pil_im.convert('RGB')), cv2.COLOR_RGB2BGR)

    # Show the processed image in a window named 'Video'
    cv2.imshow('Video', cv2_im_processed)

    # Press "0" to continue if q is pressed quit
    key = cv2.waitKey(0)
    # break out of the loop if the user presses the 'q' key
    if key == ord('q'):
        break
    

# clean up and close the window
cv2.destroyAllWindows()
