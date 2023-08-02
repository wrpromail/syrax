#!/usr/bin/env bash

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/experimental/$distribution/libnvidia-container.list | \
         sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
         sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list


sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit


docker run  --gpus all --network host -it --rm -e MODEL_PATH=/app/chatglm2-6b -v /data/chatglm2-6b:/app/chatglm2-6b -v /data/syrax/inference_serve/simple/model.py:/app/model.py -v /usr/local/lib/python3.10/dist-packages:/usr/local/lib/python3.10/site-packages 1f25d460ae70 /bin/bash
