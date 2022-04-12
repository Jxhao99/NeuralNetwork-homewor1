# 使用numpy构建两层神经网络分类器

贾翔昊 21210980038

Github地址：https://github.com/Jxhao99/NeuralNetwork-homewor1

## 运行方式

```bash
python main.py   ###主程序直接使用最新参数进行模型训练
python main.py --hidden_size XX --lr xx --reg xx --epochs xx
				##pu对指定的参数进行训练 可空
python main.py -load ###加载已经训练好的最优模型
python main.py -search  ###进行参数查找

-visual    ###上述指令后加入，-visual 可视化
```



### 参数查找：学习率，隐藏层大小，正则化强度

1. 参数查找范围如下

   ```python
   lrs = [0.025+0.002*i for i in range(5)]   ##学习率
   hidden_sizes = [40,60,80,100]   ###隐藏层
   regs = [0+0.001*i for i in range(5)]  ###正则化强度
   ```

   

2. 最佳模型的超参数为：

```python
lr = 0.033
hidden_size = 100
reg = 0

accuracy = 97.42%
```



### 可视化

```python
-visual    ###上述指令后加入，-visual 可视化，不加，默认不进行可视化
```

1. 主函数为visualize.py
2. 采用折线图对loss和accuracy可视化，采用直方图对网络每层参数进行可视化
3. 结果保存在visualize文件夹下



 ## 模型链接

链接：https://pan.baidu.com/s/1dukYnvi7F_V6d3H0kX9DqA 
提取码：6666

注意：将models文件解压，models文件夹中

