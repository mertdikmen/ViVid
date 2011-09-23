import pathfinder
import vivid
import numpy as np

# Create the reader for a list of image files
iv = vivid.ImageSource(imlist=['./media/kewell1.jpg'])
cs = vivid.ConvertedSource(iv, vivid.cv.CV_32FC3, 1.0 / 255.0)
sqs = vivid.SquaredSource(cs)

sq_frame = vivid.cvmat2array(sqs.get_frame(0))

reference_sq_frame = vivid.cvmat2array(iv.get_frame(0)).astype('float32') / 255.0
reference_sq_frame *= reference_sq_frame

print "Check: " + str(np.allclose(reference_sq_frame, sq_frame))


