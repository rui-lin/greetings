# greetings
AI to perform hands-free actions. This is work-in-progress; I'm currently working on the identity AI, to learn and recognizes humans and faces

Currently, the AI can recognize bodies by faces, track bodies and their identities (with support for hidden faces, but not body occlusion), and greet recognized people. A convenient method for adding new people to its memory has also been created.

The system uses a Haar cascade classifier and LBHP to detect and recognize faces. Foreground segmentation and contour detection, augmented by Canny edge detection, are used to detect likely bodies. An overlap of face and body allows their identities to be linked. Coarse optical flow algorithm allows the identity of the body to persist even if the face is later occluded. See below for a visual example.

New faces can be learned by standing in front of the webcam, and slowly rotating the head to capture data. The training images will be automatically cropped and aligned based on eye positions (detected via haar cascade as well).

See http://portfolio.rlin.ca/2015/09/greetings-2015.html

Setup
-----
- create virtualenv, install required python packages
  pip install -r requirements.txt

- install openCV 3.0 for your system
    I found the following link to be helpful for opencv installation on linux
    (main guide) http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/
    (as ref) http://milq.github.io/install-opencv-ubuntu-debian/
    (as ref) http://karytech.blogspot.ca/2012/05/opencv-24-on-ubuntu-1204.html

    Essentially the steps are:
    1. Install dependencies
    2. Download opencv-3.0.0 and opencv_contrib sources, extract to same location
    3. Run the following commands to install
        ```bash
        cd opencv-3.0.0 # use your dir name
        mkdir build
        cd build
        cmake -D CMAKE_BUILD_TYPE=RELEASE \
            -D CMAKE_INSTALL_PREFIX=/usr/local \
            -D INSTALL_PYTHON_EXAMPLES=ON \
            -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
            -D BUILD_EXAMPLES=ON ..
        make -j8 # run on 8 cores
        sudo make install
        sudo ldconfig

        cd ../samples/python2
        python video.py

        # symlink to allow use of opencv in our virtualenv
        ln -s /usr/local/lib/python2.7/dist-packages/cv2.so venv/lib/python2.7/site-packages/cv2.so

        # One possible error on make could be with ffmpeg complaining about
        # CodecID or AVCodecID. If so, go into the file cap_ffmpeg_impl.hpp
        # and replace the one it's complaining about with the other. Continue make.
        ```
    OpenCV 3.0 Docs: http://docs.opencv.org/master/

- python main.py # to run

Todo
----
- enhance body detection
    eg. FGDStatModel(slower) look at http://www.zoranz.net/Publications/zivkovicPRL2006.pdf
        KNN http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.62.8313&rep=rep1&type=pdf

        HOG+LBP (really good) http://www.xiaoyumu.com/s/PDF/Wang_HOG_LBP.pdf

- put expiry on add for sentences in sound queue. **
- make face go away once face detected.

- improve face recognition.
  > congealing for alignment? based on gbhuang's unsupervised alignment paper
    (http://vis-www.cs.umass.edu/papers/<iccv07alignment class="pdf"></iccv07alignment>)

- camera setting auto adjust for light??

- use camshift for moving contour thing
- + good features to track?

- deal with occluded bodies, separate faces visible. (segment foreground photos)

- try lbp cascade for face detection
- test on high # of classes of faces
- speak names

- display reconstructed image from model data (eh.. 2^(neighbours)*gridX*gridY will be very rough)
  (can display some roughly right looking one i guess. )

- merge two lbph models into one? since new training data doesn't affect old
- (extension) sharingan, highlight tiny flies in view using manifying algorithm.

- fix potential name conflict with "a b" and "a_b" in labels, disallow them.
- order imports

Done
----
- face recognition on pre-trained images [done]
- method to train for new faces on the fly [done]
- object(face) tracking [done]

- body tracking
    eg. multi bodypart. http://iris.usc.edu/outlines/papers/2006/wu-nevatia-cvpr06.pdf
    eg. get contour that contains face, mark id. if multiple, abort. 
        tracking: closest contour next frame is same person.
        can add colour histogram maybe.. nah but doesn't account rotations then.

- use face to recognize body then track it [done]

- improve face recognition
  > use haar eye detector and scale rotate translate based on that. [done]

- canny edge detection to augment contour detection [done]
- improve foreground object detection (might try just subtract static background) [done]

- store label string to int map [done]
- store training images [done]

Notes on Algorithms Used
------------------------

**HOG (Histogram of Oriented Gradients) - Object Detection**

This is a feature descriptor. Feature descriptors intend to generalize the objects they are
describing to increase robustness. This particular one uses a "global" feature vector, rather 
than a collection of local features. To deal with object translation, a fixed size sliding window 
is used, and a HOG descriptor computed for each. To deal with scaling, the image is scaled to 
multiple sizes and each searched. Rotation is dealt with by providing more data.

To compute the descriptor, 8x8 cells within the sliding window are examined, and a histogram of
the gradient angles per pixel computed, weighted by the gradient magnitude, split to nearest
bins if necessary. They are then normalized by blocks (groups of 2 by 2 cells), with 50% overlap, 
by concatenating the histograms into one large vector, and normalizing this large vector. All
the groups are concatentated to produce the final descriptor.

SVM (Support Vector Machine) is then typically used for classification.

Does not appear to work well with partial occlusions.

**__Morphological Transformations__**

**Erosion - Softly expands dark areas in a binary image.**
Works by superimposing a structuring element (eg. a 5x5 circle, line, square) on each foreground
pixel (white). If element has at least one background (black) pixel, input pixel become background.

**Dilation - softly expands white areas in a binary image**
Similar to erosion, if superimposed element has at least one foreground(white) pixel, input pixel
becomes foreground.

**Opening - reduce salt noise (extra whites).**
Same as erosion followed by dilation. Subtler effect.

**Closing - reduce pepper noise (extra blacks).**
Same as dilation followed by erosion. Subtler effect.

**__Image Preprocessing__**

Median Filter - Each pixel is replaced by median of its neighbours. The square neighbourhood,
also commonly called window, is usually 3x3 (including the pixel itself).
It's more robust than the mean filter, and is good at preserving sharp edges.
It's good for salt and pepper outlier noise, not as good for gaussian noise.
Still kinda slow. Tested use somewhat, but still slow for real time processing when #neighbours large

Gaussian Filter - Weighted average of surrounding pixels using gaussian function and distance for weight.

Bilateral Filter - Gaussian filter of space, and of pixel intensity. So nearby pixels of similar weight
are considered more. Achieves slighter better results than median, but slower, unused here.

Low pass vs high pass filter?
