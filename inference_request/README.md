该项目用于llama模型客户端测试

通过如下命令导出conda测试环境中各种包
```bash
conda env export > environment.yml
```

从制定yml文件创建conda环境
```bash
conda env create -f environment.yml
```

测试流程
1. 在test.py中指定服务地址，目前使用的是81.69.152.80，为Triton集群的CLB地址
2. 使用 ```python test.py``` 测试HTTP接口，使用 ```python test.py -i=grpc``` 测试GRPC接口

使用 ./perf_analyzer.py进行性能测试

### kubernetes 集群中运行实践