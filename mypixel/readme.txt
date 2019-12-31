pixle-anchor整体部分拆分出来的pixel部分，单独训练验证，来提高效果
evaluate 文件夹是验证代码文件夹，其中需包含需要验证数据的验证集的官方标签。
new_log_file 是训练loss记录的文档
loss.py 是anchor部分的loss，以及pixel部分的loss
model.py 包含了整体部分的网络,以及注释了anchor部分的网络，修改两部分链接处。
mydatasets.py 整体部分datasets的编写，写有注释
train.py 训练代码
eval.py 在数据集上测指标
pixel_anchor_detect.py 为pixel部分的单图推理代码