import pathfinder
import vivid

# Create the reader for a list of image files
iv = vivid.ImageSource(imlist=['./media/kewell1.jpg'])

# Source for converting to float and scaling to [0,1]
cs = vivid.ConvertedSource(iv, vivid.cv.CV_32FC3, 1.0 / 255.0)

# Source for covnerting to grayscale 
gs = vivid.GreySource(cs)

# Source for reading local binary patterns
lbps = vivid.LocalBinaryPatternSource(gs)

# Compute the local binary patterns
frame_lbp = lbps.get_lbp(0)
