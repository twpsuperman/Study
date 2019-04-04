import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf', size=14)

#设置神经网络结构参数
INPUT_NODE = 784
HIDE_NODE = 500
OUTPUT_NODE = 10
#设置神经网络学习相关参数
TRAINING_SETPS = 10000
LEARNING_RATE_BASE = 0.8 #设置基础的学习率
LEARNING_RATE_DECAY = 0.99 #学习率的衰减率
BATCH_SIZE = 100 #设置batch处理的数量
REGULARRIZATION_RATE = 0.0001 #设置正则化项前的系数
MOVING_AVERAGE_DECAY = 0.99 #设置滑动平均衰减率

#实现神经网络的前向传播函数
def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    if avg_class == None:#当没有提供滑动平均类时，直接使用当前参数的取值
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        return tf.matmul(layer1,weights2)+biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)

#定义一个训练过程
def train(mnist):
    plot_saves = []
    x = tf.placeholder(tf.float32,shape=[None,INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32,shape=[None,OUTPUT_NODE],name='y-input')
    #生成隐藏层参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,HIDE_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant([0.1],shape=[HIDE_NODE]))
    #生成输出层参数
    weights2 = tf.Variable(tf.truncated_normal([HIDE_NODE,OUTPUT_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant([0.1],shape=[OUTPUT_NODE]))
    #计算不含滑动平均的前向传播
    y = inference(x,None,weights1,biases1,weights2,biases2)
    # 定义训练轮数及相关的滑动平均类
    global_step = tf.Variable(0,trainable=False)
    #定义一个滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(LEARNING_RATE_DECAY,global_step)
    #定义一个更新变量滑动平均的操作，每次执行这个操作时，都会更新需要更新的变量
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x,variable_averages,weights1,biases1,weights2,biases2)
    #计算损失函数的交叉熵和平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y,labels = tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #计算带L2正则化的损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARRIZATION_RATE)
    regularaztion = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularaztion

    # 设置指数衰减的学习率,staircase=True;
    # 那就表明每decay_steps次计算学习速率变化，更新原始学习速率，如果是False，那就是每一步都更新学习速率
    #global_step为记录当前轮数 LEARNING_RATE_DECAY为事先设定的学习率
    #LEARNING_RATE_STEP: 是学习率更新速度, 及每LEARNING_RATE_STEP轮训练后要乘以学习率衰减系数;
    #LEARNING_RATE_DECAY为学习率衰减系数
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples / BATCH_SIZE,LEARNING_RATE_DECAY,staircase=True)
    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    # 反向传播更新参数和更新参数每一个参数的滑动平均值
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op = tf.no_op(name='train')
    #计算正确率
    correct_prediction = tf.equal(tf.arg_max(average_y,1),tf.arg_max(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    #初始化并开始进行会话
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        #准备验证数据集
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
        #准备测试数据集
        test_feed = {x:mnist.test.images,y_:mnist.test.labels}
        #循环训练神经网络
        for i in range(TRAINING_SETPS+1):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print('After %d training step(s), validation accuracy using average model is %g'%(i,validate_acc))
            dynamic_learningrate = sess.run(learning_rate, feed_dict={global_step: i})#获取动态动态学习率
            xs,ys = mnist.train.next_batch(BATCH_SIZE)#产生这一轮使用的一个batch训练的数据，并运行训练过程
            sess.run(train_op,feed_dict={x:xs,y_:ys})
            val_acc_bitch = sess.run(accuracy, feed_dict=validate_feed)#验证集准确率
            test_acc_bitch = sess.run(accuracy, feed_dict=test_feed)#测试集的准确率
            plot_save = (i,test_acc_bitch,val_acc_bitch,dynamic_learningrate)
            plot_saves.append(plot_save)
        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" % (TRAINING_SETPS, test_acc)))
    return plot_saves

def plot_accuracy(plot_saves):
    m = len(plot_saves)
    x_epoach = []#保存训练次数的列表
    y_test_acc = []#保存测试集准确率的列表
    y_val_acc = []#保持验证集准确率列表
    dynamic_learning_rate = []#保存动态学习率的列表
    for i in range(m):
        x_epoach.append(plot_saves[i][0])
        y_test_acc.append(plot_saves[i][1])
        y_val_acc.append(plot_saves[i][2])
        dynamic_learning_rate.append(plot_saves[i][3])
    plt.figure(1)
    plt.plot(x_epoach,y_test_acc,'m',label = 'test acc')
    plt.plot(x_epoach,y_val_acc,'r',label='val acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('测试集的正确率', fontproperties=font)  # 绘制图形的标题
    plt.legend()  # 显示图形的图例
    plt.figure(2)
    plt.plot(x_epoach, dynamic_learning_rate, 'r', label='learning_rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning_rate')
    plt.title('动态学习率曲线', fontproperties=font)  # 绘制图形的标题
    plt.legend()#显示图形的图例
    plt.show()

def main(argv=None):
    mnist = input_data.read_data_sets(r"D:\唐文平学习资料\神经网络\TensorFlow实战Google深度学习框架\05_手写数字识别\MNIST_data",one_hot=True)
    plot_saves = train(mnist)
    plot_accuracy(plot_saves)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("总花费时间为："+str((end-start)/60)+'min')