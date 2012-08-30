import pathfinder
import vivid

import matplotlib.pyplot as plt

plt.ion()

video_file = '../media/Standard_MPEG-2.mpeg'

fv = vivid.FileVideo(video_file)

frame = fv.get_frame(10)

plt.imshow(frame)

