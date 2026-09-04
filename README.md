# :computer: :school: ROS Summer :school: :computer:
## :camera: Mobile Robot Lane Following with Computer Vision :camera:

Vision-based lane following for a model car, built on **ROS Kinetic** + **OpenCV** in **Python 2.7**.

A camera streams frames over the ROS network; a vision node thresholds the image, finds the
lane-marking blobs, fits a line through their centroids, and turns the offset between that line
and the image center into a steering angle that is published back to the car's motor controller.

This repository is a backup of a summer project on a leader/follower mobile robot pair, and holds
the **leader** robot's code. In the client/server model used, the **robot is the client**; any
machine running ROS with the right configuration and these scripts can act as the **server**
(ROS master). After calibrating the camera angle and the wheel-degree mapping, the robot drove
straight and curved sections of a taped highway on a table.

---

## Table of Contents

- [Quick Start](#quick-start)
- [The Car](#the-car)
- [How It Works](#how-it-works)
- [Repository Layout](#repository-layout)
- [Skills & Techniques](#skills--techniques)
- [Tools & Libraries](#tools--libraries)
- [ROS Interface](#ros-interface-topics--services)
- [Network Protocols](#network-protocols)
- [Known Issues](#known-issues)
- [Bibliography](#closed_book-bibliography-closed_book)

---

## Quick Start

### Requirements

| Component | Version / Notes |
|---|---|
| OS | Ubuntu 16.04 (ROS Kinetic target) |
| ROS | Kinetic Kame |
| Python | 2.7 (the code uses `print` statements and Python-2 division) |
| OpenCV | 3.x (`cv2.findContours` returns 3 values; `cv2.bgsegm` used) |
| Python packages | `numpy`, `matplotlib`, `opencv-python`, `cv_bridge` (from `ros-kinetic-cv-bridge`) |
| Robot | [AutoModelCar](https://github.com/AutoModelCar)-style platform exposing `/manual_control/*` topics — see below |

### The car

The vehicle is an [AutoModelCar](https://github.com/AutoModelCar)-family platform — the 1:10-scale
autonomous model car developed at **Freie Universität Berlin** for teaching and research. The
relevant traits for this code:

| Part | What it means here |
|---|---|
| **Odroid** onboard computer | Runs the ROS nodes on the car; the vision node can also run off-board over the network |
| **Arduino** low-level controller | Consumes `/manual_control/steering` and `/manual_control/speed` and drives the servo and motor |
| **Intel RealSense** camera | The source of `app/camera/color/image_raw` — the color stream this project republishes as `Image` |
| 1:10 scale chassis | Sets the geometry behind the `29.35/4.8` steering calibration constant |

Steering is a servo angle where **90 is straight ahead**; speed is negative to drive forward on
this platform (`-100`), and `0` stops. Upstream reference code and hardware notes live in the
[AutoModelCarWiki](https://github.com/AutoModelCar/AutoModelCarWiki/wiki) and
[catkin_ws_user](https://github.com/AutoModelCar/catkin_ws_user) repositories.

### Network setup (robot ↔ server)

On **every** machine, point at the same ROS master and advertise your own reachable IP:

```bash
ifconfig                                        # find your IP
export ROS_MASTER_URI=http://192.168.50.200:11311   # the master's address
export ROS_IP=192.168.50.246                        # this machine's address
```

Start the master on the server machine:

```bash
roscore
```

### Running the pipeline

1. **Publish camera frames.** Either from a local webcam:

   ```bash
   python publisher_image.py          # /dev/video0 -> ROS topic "Image" (bgr8)
   ```

   or, on the car, by re-publishing the compressed camera stream as raw:

   ```bash
   rosrun image_transport republish compressed \
       in:=app/camera/color/image_raw raw out:=Image
   ```

2. **Run the lane follower** (subscribes to `Image`, drives the car):

   ```bash
   python subscriber_image.py
   ```

### Trying the vision offline (no ROS needed)

```bash
python car_main.py                # full pipeline on a still image (prueba_1.jpg)
python car_centroid.py            # centroid + regression on camino_art_4.png
python centroid_opencv.py         # HSV white mask -> largest-contour centroid
python hough_lines_1.py           # Canny + Hough line overlay
python lane_lines_detection.py    # RGB white/yellow mask on cropped frame
python thresholding_gray.py       # global vs adaptive thresholding comparison
python gray_scale.py              # minimal grayscale load/display
```

### ROS "hello world" sanity checks

```bash
python talker.py    &   # publishes std_msgs/String on "prueba"
python listener.py      # subscribes and logs it

python add_two_ints_server.py   # advertises the "add_two_ints" service
python add_two_ints_client.py 3 4
```

---

## How It Works

The full control loop lives in `subscriber_image.py` (`car_main.py` is the same pipeline run
against a still image, for tuning). Each incoming frame goes through:

```
sensor_msgs/Image
   │  cv_bridge: imgmsg_to_cv2 (bgr8)
   ▼
Grayscale  ──►  Invert (bitwise_not)      # lane markings become the bright blobs
   ▼
Crop to bottom 40% of frame               # only the road right in front of the car
   ▼
Gaussian blur 15x15                       # kill sensor noise
   ▼
Binary threshold @ 160                    # (offline variant uses Otsu)
   ▼
Morphological CLOSE, 9x9 kernel           # fill holes in the markings
   ▼
findContours (RETR_TREE, CHAIN_APPROX_NONE)
   ▼
Filter contours by area > 110 px, sort by area
   ▼
Image moments -> centroid (cx, cy) per blob
   ▼
Sort centroids by x, drop the right-most one
   ▼
Least-squares linear regression  y = m·x + b   # the lane line
   ▼
Midpoint between the lane line and the right-most centroid = target point
   ▼
Right triangle from image-bottom-center to target -> angle = arcsin(a/c)
   ▼
Scale by the calibrated 29.35/4.8 wheel factor, offset around 90° neutral
   ▼
std_msgs/Int16 -> /manual_control/steering  (+ constant speed on /manual_control/speed)
```

Steering is clamped to the `0..180` servo range, snapping back to `90` (straight) if the
computed angle leaves it. If the regression, the line drawing, or the intercept solve raises
(degenerate blob set, vertical line, division by zero), the node publishes **speed 0** and
returns — a fail-safe stop rather than a stale command.

---

## Repository Layout

### ROS nodes

| File | Role |
|---|---|
| `subscriber_image.py` | **Main node.** Subscribes to `Image`, runs the vision pipeline, publishes steering + speed. |
| `publisher_image.py` | Captures from `cv2.VideoCapture(0)` at 10 Hz and publishes `sensor_msgs/Image` (bgr8) on `Image`. |
| `ros_test.py` | Earlier camera publisher, publishing on `VideoRaw` (unthrottled, `RGB8`). |
| `talker.py` | ROS tutorial publisher — `std_msgs/String` on `prueba` at 10 Hz. |
| `listener.py` | ROS tutorial subscriber for `prueba`. |
| `add_two_ints_server.py` | ROS **service** server for `beginner_tutorials/AddTwoInts`. |
| `add_two_ints_client.py` | Matching service client (`ServiceProxy` + `wait_for_service`). |

### Offline vision experiments

| File | What it explores |
|---|---|
| `car_main.py` | The complete pipeline on a still image — the tuning bench for `subscriber_image.py`. |
| `car_centroid.py` | Centroid extraction, blob exclusion by x-range, regression, matplotlib plot to `grafica.png`. |
| `centroid_opencv.py` | HSV white-range mask, erode/dilate denoising, largest-contour centroid. |
| `hough_lines_1.py` | Canny edges + `cv2.HoughLines`, drawn with `addWeighted` blending. |
| `hough_lines_2.py` | Hough variant with a tighter blur and per-line incremental display. |
| `lane_lines_detection.py` | RGB white/yellow color mask via `inRange` + `bitwise_and`. |
| `thresholding_gray.py` | Global vs adaptive-mean vs adaptive-Gaussian thresholding, side by side. |
| `gray_scale.py` | Minimal grayscale load/display. |

### Assets

- **Test images:** `carretera*.jpg/png`, `camino_art_[1-4].png`, `white_test*.jpg`, `lane_line*`, `prueba_1.jpg`
- **Outputs:** `grafica.png` (regression plot), `roslib` (PostScript diagram of the ROS graph)
- **Strays:** `.fuse_hidden*` — orphaned FUSE files holding earlier blob-detector and
  filter/background-subtraction experiments (`SimpleBlobDetector`, `bgsegm.createBackgroundSubtractorMOG`).

---

## Skills & Techniques

**Image preprocessing**
- Color-space conversion — `cvtColor` (BGR↔GRAY, RGB↔HSV)
- Image inversion — `bitwise_not`
- ROI cropping via numpy slicing (bottom 40–50% of the frame)
- Gaussian blur, median blur, bilateral filtering

**Segmentation**
- Global binary thresholding
- Otsu's automatic threshold
- Adaptive thresholding (mean and Gaussian)
- Color-range masking — `inRange` in RGB and HSV, combined with `bitwise_and`

**Morphology**
- Erosion, dilation
- Closing (`MORPH_CLOSE`) to seal gaps in lane markings
- Structuring elements via `np.ones` and `getStructuringElement`

**Feature extraction**
- Contour finding (`RETR_TREE`, `CHAIN_APPROX_NONE` / `CHAIN_APPROX_SIMPLE`)
- Contour area filtering and area-based sorting
- **Image moments** → centroid as `cx = m10/m00`, `cy = m01/m00`
- Blob detection (`SimpleBlobDetector` with area filtering)
- Canny edge detection
- **Hough line transform** with polar→Cartesian line reconstruction
- Background subtraction (MOG)

**Math & control**
- Least-squares **linear regression** implemented by hand from the summation form
- Right-triangle trigonometry — `arcsin(a/c)` for the heading angle, radians→degrees
- Servo calibration: empirical `29.35/4.8` degrees-per-unit factor with a `0.2` damping gain
- Angle clamping to the servo's valid range with a neutral-90° fallback
- Exception-guarded fail-safe stop on degenerate geometry

**ROS**
- Publisher / subscriber nodes, and both in a single script
- Service server / client (`rospy.Service`, `ServiceProxy`, `wait_for_service`)
- Anonymous node naming, `rospy.Rate` loop timing, `rospy.spin()`
- `cv_bridge` conversion between `sensor_msgs/Image` and OpenCV arrays
- Distributed master/node configuration across machines
- `image_transport` republishing (compressed → raw)

**Visualization & debugging**
- `imshow` / `waitKey` inspection of every pipeline stage
- Drawing primitives — `line`, `circle`, `drawContours`, `rectangle`, `addWeighted`
- matplotlib subplot grids and `savefig` for the regression plot

---

## Tools & Libraries

| Tool | Used for |
|---|---|
| **ROS Kinetic** | Robot middleware — the node graph, transport, and naming |
| **rospy** | Python ROS client library |
| **OpenCV 3.x (`cv2`)** | All image processing, and the debug GUI (HighGUI) |
| **`cv2.bgsegm`** | `opencv_contrib` background-subtraction module |
| **cv_bridge** | `sensor_msgs/Image` ↔ numpy/OpenCV conversion |
| **NumPy** | Array math, kernels, regression arithmetic, `arcsin` |
| **matplotlib** | Centroid scatter plots and regression figures |
| **`operator.itemgetter`** | Sorting the centroid dictionary by x |
| **`std_msgs`** | `String`, `Int16` message types |
| **`sensor_msgs`** | `Image` message type |
| **`image_transport`** | `republish` for the compressed camera stream |
| **`roscore` / `rosrun`** | Master startup and node launching |
| **`ifconfig`** | Finding the IP for `ROS_IP` / `ROS_MASTER_URI` |
| **Git / GitHub** | Version control (see the `contours` feature branch merge) |

---

## ROS Interface (Topics & Services)

### Published topics

| Topic | Type | Publisher | Notes |
|---|---|---|---|
| `/manual_control/steering` | `std_msgs/Int16` | `subscriber_image.py` | Servo angle, `0..180`, `90` = straight |
| `/manual_control/speed` | `std_msgs/Int16` | `subscriber_image.py` | `-100` to drive, `0` to stop |
| `/manual_control/stop_start` | `std_msgs/Int16` | `subscriber_image.py` | Enable/disable line (declared; currently commented out) |
| `Image` | `sensor_msgs/Image` | `publisher_image.py` | `bgr8`, 10 Hz, queue size 10 |
| `VideoRaw` | `sensor_msgs/Image` | `ros_test.py` | Legacy camera topic |
| `prueba` | `std_msgs/String` | `talker.py` | Tutorial topic (replaces `chatter`) |

### Subscribed topics

| Topic | Type | Subscriber |
|---|---|---|
| `Image` | `sensor_msgs/Image` | `subscriber_image.py` |
| `prueba` | `std_msgs/String` | `listener.py` |

### Services

| Service | Type | Server / Client |
|---|---|---|
| `add_two_ints` | `beginner_tutorials/AddTwoInts` | `add_two_ints_server.py` / `add_two_ints_client.py` |

### Nodes

`listener` (the lane follower, anonymous), `talker` / `VideoPublisher` (camera publishers),
`add_two_ints_server`.

---

## Network Protocols

ROS's own transport stack is what carries every message in this project:

| Layer | Protocol | Where it shows up |
|---|---|---|
| **Master / naming** | **XML-RPC over HTTP**, default port **11311** | `ROS_MASTER_URI=http://192.168.50.200:11311` — nodes register and look up peers here |
| **Node API** | **XML-RPC over HTTP**, ephemeral ports | Each node runs its own XML-RPC server for negotiation and parameter callbacks |
| **Message transport** | **TCPROS** (ROS message stream over TCP) | Default for `rospy.Publisher` / `rospy.Subscriber` — carries the `Image` and `Int16` traffic |
| | **UDPROS** (over UDP) | Available alternative for lossy/low-latency streams; not used here |
| **Services** | **TCPROS** request/response | `add_two_ints` — `ServiceProxy` opens a TCP connection to the server node |
| **Parameter server** | **XML-RPC over HTTP** (on the master) | Backs `rospy` parameter access |
| **Addressing** | **IPv4 / TCP / UDP**, DNS or raw IP | `ROS_IP` pins the address a node advertises to peers; the setup uses literal IPs on the `192.168.50.0/24` LAN |
| **Image streaming** | `image_transport` compressed vs. raw | `republish compressed in:=… raw out:=Image` — compressed frames over the wire, decoded to raw for the pipeline |

**Practical notes**

- The master and every node must be **mutually reachable**; because the master hands out
  addresses that nodes then connect to directly, a wrong `ROS_IP` produces nodes that register
  fine but never exchange data.
- Ephemeral node ports mean a permissive firewall on the robot LAN, not just port 11311.
- `queue_size=1` on the steering/speed publishers deliberately drops stale commands rather than
  buffering them — the newest control value is the only one that matters.
- Raw `sensor_msgs/Image` frames are large; publishing the **compressed** transport and
  republishing to raw at the consumer is what keeps the Wi-Fi link usable.

---

## Known Issues

These are the remaining rough edges in the tree:

- Bare `except:` clauses in `subscriber_image.py` swallow every error, including typos; they do
  at least fail safe by publishing speed 0.
- Python 2 only — `print` statements, integer division, and `xrange` all need porting for
  Python 3 / ROS 2.
- `add_two_ints_client.py` / `add_two_ints_server.py` need the `beginner_tutorials` package
  (with its `AddTwoInts.srv`) built and on `PYTHONPATH`; they are ROS tutorial code, not part of
  the car pipeline.
- The `.fuse_hidden*` files are FUSE leftovers, not intentional source; they are kept because
  they hold blob-detector experiments not represented elsewhere.

### Recently fixed

- `car_main.py` no longer carries the unresolved merge-conflict markers from commit `81d320b`;
  the calibrated angle branch (matching `subscriber_image.py`) was kept.
- `add_two_ints_client.py` has its function body and `beginner_tutorials.srv` import restored.
- `ros_test.py` now passes the encoding to `cv2_to_imgmsg` instead of to `publish()`, and its
  10 Hz `rate.sleep()` is active.

---

##  :closed_book: Bibliography :closed_book:

The references that shaped this project, grouped by the part of the pipeline they informed.
Duplicates from the original flat list have been merged; links verified 2026-09-04, with
:warning: marking dead originals (an archived copy is linked where one exists).

### Reference projects — vision-guided model cars

* [Autonomous Racing Robot With an Arduino, a Raspberry Pi and a Pi Camera](https://becominghuman.ai/autonomous-racing-robot-with-an-arduino-a-raspberry-pi-and-a-pi-camera-3e72819e1e63) — the closest analogue to this build: crop, threshold, centroid, steer
* Voiture de Course Autonome avec une Arduino, une Raspberry Pi et une Pi Camera — French original of the above :warning: (`enstar.ensta-paristech.fr` is gone, no archived copy; read the English version)
* [AutoModelCar](https://github.com/AutoModelCar) — the 1:10-scale platform family this car belongs to (Freie Universität Berlin; Odroid + Arduino + Intel RealSense), source of the `/manual_control/*` topics and the `app/camera/color/image_raw` stream
* [How to build a self-driving car in one month](https://medium.com/@maxdeutsch/how-to-build-a-self-driving-car-in-one-month-d52df48f5b07)
* [naokishibuya/car-finding-lane-lines](https://github.com/naokishibuya/car-finding-lane-lines) — color masking + Hough lane finding
* [galenballew/SDC-Lane-and-Vehicle-Detection-Tracking](https://github.com/galenballew/SDC-Lane-and-Vehicle-Detection-Tracking)
* [CRM-UAM/VisionRace](https://github.com/CRM-UAM/VisionRace)
* [Line Following Robot OpenCV](https://www.youtube.com/watch?v=ZC4VUt1I5FI) — video walkthrough
* [Building a Line Following BeagleBone Robot with openCV](http://web.archive.org/web/20240414092134/https://einsteiniumstudios.com/beaglebone-opencv-line-following-robot.html) :warning: (archived)

### ROS

* [ROS Tutorials](http://wiki.ros.org/ROS/Tutorials) — source of `talker.py`, `listener.py`, and the `add_two_ints` service pair
* [rospy wiki](http://wiki.ros.org/rospy) · [rospy Tutorials](http://wiki.ros.org/rospy/Tutorials)
* [std_msgs](http://wiki.ros.org/std_msgs) — `String` and `Int16`, the message types used here
* [ROS Kinetic: Publisher and Subscriber in Python](https://www.intorobotics.com/ros-kinetic-publisher-and-subscriber-in-python/)
* [Writing a ROS node with both a publisher and subscriber?](https://stackoverflow.com/questions/40508651/writing-a-ros-node-with-both-a-publisher-and-subscriber) — the pattern `subscriber_image.py` uses
* [Publisher/subscriber in one python script](https://answers.ros.org/question/107326/publishersubscriber-in-one-python-script/)
* [Multiple subscribers and single publisher in one python script?](https://answers.ros.org/question/232216/multiple-subscribers-and-single-publisher-in-one-python-script/)

### OpenCV — general

* [OpenCV-Python Tutorials](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)
* [Getting Started with Images](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html#py-display-image)
* [Basic Operations on Images](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html)
* [Drawing Functions in OpenCV](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_drawing_functions/py_drawing_functions.html#drawing-circle) · [drawing functions reference](https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#line)
* [Miscellaneous Image Transformations](https://docs.opencv.org/3.4.0/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57)
* [Building a Pokedex in Python: Getting Started](https://www.pyimagesearch.com/2014/03/10/building-pokedex-python-getting-started-step-1-6/)
* [Camera Calibration](http://web.archive.org/web/20210507124408/https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html) :warning: (archived)
* [Pixel](https://en.wikipedia.org/wiki/Pixel) · [sensor resolution diagram](https://en.wikipedia.org/wiki/Pixel#/media/File:Sensoraufl%C3%B6sungen.svg)

### Filtering & smoothing

* [Smoothing Images](http://web.archive.org/web/20210412061431/https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html) :warning: (archived)
* [Gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur) — the 15×15 kernel in the pipeline
* [Bilateral Filtering](http://eric-yuan.me/bilateral-filtering/) — used in `car_centroid.py`
* [Image Gradients](https://docs.opencv.org/3.1.0/d5/d0f/tutorial_py_gradients.html)

### Thresholding & morphology

* [Image Thresholding](https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html) — global, Otsu, and adaptive, all compared in `thresholding_gray.py`
* [Threshold (Umbralización)](https://sites.google.com/site/cg05procesamientodeimagenes/home/threshold-umbralizacion)
* [Morphological Transformations](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html) — the 9×9 `MORPH_CLOSE` that seals the lane markings
* [Inverting an image in Python with OpenCV](https://stackoverflow.com/questions/19580102/inverting-image-in-python-with-opencv) — the `bitwise_not` step that makes markings the bright blobs
* [Filling holes in an image using OpenCV](https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/)

### Contours, moments & blobs

* [Contours: Getting Started](https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html)
* [Contour Features](https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html)
* [Structural Analysis and Shape Descriptors](https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gacb413ddce8e48ff3ca61ed7cf626a366) — `moments()` and `contourArea()`
* [Image moment](https://en.wikipedia.org/wiki/Image_moment) — the `cx = m10/m00`, `cy = m01/m00` centroid at the heart of the controller
* [Blob detection](https://en.wikipedia.org/wiki/Blob_detection)
* [Blob Detection Using OpenCV (Python, C++)](https://www.learnopencv.com/blob-detection-using-opencv-python-c/)
* [How to select a bounding box (ROI) in OpenCV](https://www.learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/)
* [cvblob](https://code.google.com/archive/p/cvblob/) · [opencvblobslib](http://opencvblobslib.github.io/opencvblobslib/)

### Edges & Hough transform

* [Hough Line Transform](http://web.archive.org/web/20210301012812/https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html) :warning: (archived)
* [Hough transform](https://en.wikipedia.org/wiki/Hough_transform)
* [Hough Lines Transform Explained](http://tomaszkacmajor.pl/index.php/2017/06/05/hough-lines-transform-explained/) — the polar→Cartesian reconstruction in `hough_lines_1.py`
* [Line Detection by Hough Transformation (PDF)](http://web.archive.org/web/20221205194047/https://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/HoughTrans_lines_09.pdf) :warning: (archived)

### Background subtraction

* [Background Subtraction](https://docs.opencv.org/3.1.0/db/d5c/tutorial_py_bg_subtraction.html)
* [bgslibrary](https://github.com/andrewssobral/bgslibrary#bgslibrary)
* [simple_vehicle_counting](https://github.com/andrewssobral/simple_vehicle_counting)

### Math — regression & trigonometry

* [Linear regression](https://en.wikipedia.org/wiki/Linear_regression) — the least-squares fit implemented by hand from the summation form
* [Linear Regression Example (scikit-learn)](http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html)
* [Triángulo rectángulo — razones trigonométricas](https://www.euston96.com/triangulo-rectangulo/#Razones_trigonometricas) — the right triangle behind the steering angle
* [numpy.arcsin](https://numpy.org/doc/stable/reference/generated/numpy.arcsin.html) — link updated; the old `docs.scipy.org` path is dead
* [numpy.asarray](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html) — link updated; the old `docs.scipy.org` path is dead

### Python 2 & plotting

* [Handling Exceptions](https://docs.python.org/2/tutorial/errors.html)
* [range()](https://docs.python.org/2/library/functions.html#range)
* [Dictionaries](https://www.python-course.eu/dictionaries.php)
* [sorted()](https://www.programiz.com/python-programming/methods/built-in/sorted) · [How do I sort a dictionary by value?](https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value) — how the centroids get ordered by x
* [matplotlib tutorials](https://matplotlib.org/tutorials/index.html) · [savefig()](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html) — writes `grafica.png`

### Project notes

* pista — *CarroAutonomo* track/design notes (Google Doc, **access-restricted**: the original link requires sign-in and is not publicly readable)
