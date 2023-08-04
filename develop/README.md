使用 simple 方式
1. 构建包含 codegeex2 依赖的 docker 镜像, 需确保镜像中含有 wget 工具
2. 下载模型物料
3. docker 运行 1中的容器，并且挂载模型物料（transformers 库缓存文件）验证模型可运行
4. 编写 docker-compose 或 kubernetes deployment 声明文件
5. 编写 human-eval 仓库的适配调用脚本并获取 human-eval 结果
6. 编写自定义推理结果收集代码

