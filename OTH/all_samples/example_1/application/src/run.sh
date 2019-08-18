#!/bin/bash
# This is the container entry-point
echo
echo ----------------------------
echo CONTAINER RUN SCRIPT STARTED
echo ----------------------------
echo
cd /application
python tf-mnist.py
