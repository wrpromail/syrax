#!/usr/bin/env bash

# 如果要在 docker 中使用 GPU，需要安装 nvidia-container-toolkit
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/experimental/$distribution/libnvidia-container.list | \
         sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
         sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 在自己的测试环境临时运行测试环境
# 可以将模型目录挂载到容器中，避免每次都需要进行下载
# 可以将宿主机的 python3 site 目录也挂载进去，在容器中不再安装依赖，当然这个方式是不推荐的
# 注意安装上述 nvidia-container-toolkit 在执行 docker run 命令时需指定 --gpus all
docker run  --gpus all --network host -it --rm -e MODEL_PATH=/app/chatglm2-6b -v /data/chatglm2-6b:/app/chatglm2-6b -v /data/syrax/inference_serve/simple/model.py:/app/model.py -v /usr/local/lib/python3.10/dist-packages:/usr/local/lib/python3.10/site-packages 1f25d460ae70 /bin/bash
