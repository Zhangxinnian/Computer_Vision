'''
import tensorflow as tf
import os

model_name = './ICDAR_0.7.pb'

def create_graph():
    with tf.gfile.FastGFile(model_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

create_graph()
tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]

for tensor_name in tensor_name_list:
    print(tensor_name,'\n')
'''
'''
import tensorflow as tf

model_path = "./ICDAR_0.7.pb"

with tf.gfile.FastGFile(model_path,'rb')as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	tf.import_graph_def(graph_def,name='')

	# tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
	# with open('tensor_name_list_pb.txt','a')as t:
	# 	for tensor_name in tensor_name_list:
	# 		t.write(tensor_name+'\n')
	# 		print(tensor_name,'\n')
	# 	t.close()
	with tf.Session()as sess:
		op_list = sess.graph.get_operations()
		with open("model里面张量的属性.txt",'a+')as f:
			for index,op in enumerate(op_list):
				f.write(str(op.name)+"\n")                   #张量的名称
				f.write(str(op.values())+"\n")
'''
'''
import tensorflow as tf
import os
os.makedirs('./pb',exist_ok=True)
# 加载模型
output_graph_def = tf.GraphDef()
with open('./ICDAR_0.7.pb', "rb") as f:
    output_graph_def.ParseFromString(f.read())
    tensors = tf.import_graph_def(output_graph_def, name="")
sess = tf.Session()
sess.run(tf.global_variables_initializer())
graph = tf.get_default_graph()

# 生成tensorboard文件
file_writer = tf.summary.FileWriter('./pb')
file_writer.add_graph(graph)
file_writer.flush()

# 打印模型中所有的操作
op = graph.get_operations()
for i, m in enumerate(op):
   print('op{}:'.format(i), m.values())
'''

import tensorflow as tf

class python_model():
    def __init__(self,model_path):
        # 读取模型
        output_graph_def = tf.GraphDef()
        # 打开.pb模型
        with open(model_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tensors = tf.import_graph_def(output_graph_def, name="")

        self.__sess = tf.Session()
        self.__sess.run(tf.global_variables_initializer())
        graph = tf.get_default_graph()
        self.__input = graph.get_tensor_by_name("image:0")
        self.__output = graph.get_tensor_by_name("output/masks:0")

    def inference(self,input):
        output = self.__sess.run(self.__output, feed_dict={self.__input: input})
        return output

