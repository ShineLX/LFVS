# This line defines the base image of the container.
#
# Our example container is derived from nVidia's image
# 'tensorflow:18.06-py3', the code stands
# for container version 18.06 (nVidia's internal version),
# which contains tensorflow set up for python3
#
#FROM nvcr.io/nvidia/tensorflow:18.06-py3
#FROM rohitdockeryes/nvd-tf-cv2
FROM ccu.uni-konstanz.de:5000/rohit.choudhary/nvd-tf-cv2

#
# This is the maintainer of the container, referenced
# by e-Mail address.
#
MAINTAINER rohit.choudhary@uni-konstanz.de
#RUN apt-get update
#RUN apt-get install python-opencv


#
# This is the first line which tells us how this container
# differs from the base image.
#
# In this case, we copy the subdirectory "src" from
# the directory containing the Dockerfile into the
# directory "/application" of the container image.
#
COPY src /application
