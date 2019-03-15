import tensorflow as tf

#配置神经网络的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
#第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5
#第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5
#全连接层节点数
FC_SIZE = 512

#定义卷积神经网络的前向传播过程
def inference(input_tensor,train,regularizer):
    input_tensor = tf.reshape(input_tensor,[-1,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS])
    with tf.variable_scope("layer1-conv1"):
        conv1_weights = tf.get_variable('weight',[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('biases',[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        #实现第一层的前向传播
        #使用过滤器的尺寸为5*5，深度为32，步长为1，使用全零填充
        #输出为(28,28,32)
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    #实现第二层的前向传播（最大池化层)
    #使用过池化层滤器的尺寸为2*2，步长为2，使用全零填充
    #输出为（14,14,32）
    with tf.variable_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    #实现第三层前向传播
    #输出为（14,14,64）
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable('weights',[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('biases',[CONV2_DEEP],initializer=tf.constant_initializer((0.0)))
        #过滤器的尺寸(5,5),深度64，步长为1，使用零填充
        conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    #实现第四层池化层的前向传播过程
    #输出为（7,7,64）
    with tf.variable_scope("layer4_pool2"):
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    #将第四层池化层的输出转化为第五层全连接层的输出格式
    #第五层的全连接层需要的输入格式是向量，将7*7*64的矩阵拉直成一个向量
    pool_shape = pool2.get_shape().as_list()#得到第四层矩阵输出的维度
    #pool_shape[0]为batch中个数，转化为向量
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    #通过tf.reshape函数将第四层的输出变为一个batch向量
    reshaped = tf.reshape(pool2,[-1,nodes])
    #声明第五层全连接的变量并实现前向传播
    with tf.variable_scope("layer5_fc1"):
        fc1_weights = tf.get_variable('weights',[nodes,FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        #只有全连接层才加入正则化
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable('biases',[FC_SIZE],initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1,0.8)
    #声明第六层全连接的变量并实现前向传播
    with tf.variable_scope("layer6_fc2"):
        fc2_weights = tf.get_variable('weights',[FC_SIZE,NUM_LABELS],initializer = tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases = tf.get_variable('biases',[NUM_LABELS],initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1,fc2_weights) + fc2_biases
    return logit


