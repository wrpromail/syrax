#!/usr/bin/env bash

# 需要配置的环境变量
# TARGET_IP
# TARGET_PORT
# PROTOCOL

# 运行 docker 容器加载测试环境
docker run -it --rm -v .:/app --network host cs-ai.tencentcloudcr.com/triton/llama-test:local-dev-0725 /bin/bash