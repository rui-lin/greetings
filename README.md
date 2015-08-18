# greetings
AI to recognize and greet people as they walk by

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
- face recognition on pre-trained images (done)
- method to train for new faces on the fly (done)
- object(face) tracking
- (idea) canny edge detection to augment contour detection
- try lbp cascade for face detection
- test on high # of classes of faces
- speak names

- store label string to int map (done)
- store training images (done)
- display reconstructed image from model data (eh.. 2^(neighbours)*gridX*gridY will be very rough)
  (can display some roughly right looking one i guess. )

- merge two lbph models into one? since new training data doesn't affect old
- (extension) sharingan, highlight tiny flies in view using manifying algorithm.

- fix potential name conflict with "a b" and "a_b" in labels, disallow them.
- order imports

Notes on Algorithms Used
------------------------

HOG (Histogram of Oriented Gradients) - Object Detection

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