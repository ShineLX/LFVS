#!/bin/bash
#
# This script illustrates how to generate kubectl configuration
# files from templates by substituting environment variables.
#
# Input: your Kubernetes username in the environment variable
#        KUBERNETES_USER. A suggestion is to set that in your
#        bashrc login script to not forget.
#        

# Auto-generate docker user name and name prefix for Kubernetes objects 
set -a
DOCKER_USER=${KUBERNETES_USER}
KUBERNETES_USER_PREFIX=${KUBERNETES_USER//./-}

# Some feedback about what we are doing
echo Creating config files from templates.
echo   Kubernetes user: ${KUBERNETES_USER}
echo   Docker user    : ${DOCKER_USER}
echo   Name prefix    : ${KUBERNETES_USER_PREFIX}
echo
echo Files:

# Substitute environment vars in all templates
# to generate Kubernetes yaml files
for f in templates/*.template
do
  base_name=`basename ${f%.template}`
  echo "  " $base_name
  envsubst < $f > $base_name.yaml
done

