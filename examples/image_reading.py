import pathfinder
import vivid

from matplotlib.pyplot import imshow, show, gray

# Create the reader for a list of image files
iv = vivid.ImageSource(imlist=['./media/kewell1.jpg'])

# Source for converting to float and scaling to [0,1]
cs = vivid.ConvertedSource(iv, vivid.cv.CV_32FC3, 1.0 / 255.0)

# Source for covnerting to grayscale 
gs = vivid.GreySource(cs)

# Source for resizing
ss = vivid.ScaledSource(gs, 0.5)

# Get the first image
frame = cs.get_frame(0)

# Get the grayscale version of the first image
frame_gray = gs.get_frame(0)

# Get the scaled version of the grayscale image
frame_gray_scaled = ss.get_frame(0)

# Convert into numpy arrays
fr = vivid.cvmat2array(frame)
fr_gray = vivid.cvmat2array(frame_gray)
fr_gray_scaled = vivid.cvmat2array(frame_gray_scaled)
