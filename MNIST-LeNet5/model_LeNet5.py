import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import mnist_inference_LeNet5
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf', size=14)

#定义神经网络相关参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01#基础学习率
LEARNING_RATE_DECAY = 0.99#学习衰减率
REGULARIZATION_RATE = 0.0001
TRAINING_STEP = 5000#学习训练的轮数
MOVING_AVERAGE_DECAY = 0.99#移动加权平均

#维度转换
def mnist_reshape(_batch):
    batch = np.reshape(_batch, [-1, 28, 28])
    return batch

#定义训练过程
def train(mnist):
    plot_saves = [] #保存训练过程的参数，方便后面绘图
    # 命名空间
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32,[None,mnist_inference_LeNet5.IMAGE_SIZE,mnist_inference_LeNet5.IMAGE_SIZE],name='x-input')
        y_ = tf.placeholder(tf.float32,[None,mnist_inference_LeNet5.OUTPUT_NODE],name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference_LeNet5.inference(x,False,regularizer)
    global_step = tf.Variable(0,trainable=False)#定义训练轮数
    #定义损失函数、学习率、滑动平均操作以及训练过程
    variables_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)#定义一个滑动平均类
    variables_averages_op = variables_averages.apply(tf.trainable_variables())#对所有的训练变量执行滑动平均操作

    with tf.name_scope('loss'):
        #labels的维度是一维(batch_size,) logits的维度是(batch_size,10)
        #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))  # 定义损失函数
        tf.summary.scalar('loss',loss)#数据多用上边定义的函数,loss只有一个值调用没有意义

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY,staircase=True)
    with tf.name_scope('train'):
        # 优化损失函数
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    # 反向传播更新参数和更新参数每一个参数的滑动平均值
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op = tf.no_op(name='train')

    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_prediction"):
            #计算正确率
            correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            tf.summary.scalar('accuracy', accuracy)

    # 准备测试集数据
    xtest,ytest = mnist.test.next_batch(1000)
    x_test_reshaped = mnist_reshape(xtest)
    test_feed = {x:x_test_reshaped,y_:ytest}
    #初始化并开始进行会话
    init_op = tf.global_variables_initializer()

    # 合并所有的summary
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init_op)
        writer = tf.summary.FileWriter('logs/', sess.graph)
        for i in range(TRAINING_STEP+1):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            xs_reshape = mnist_reshape(xs)
            # #将输入的训练数据格式调整为一个4维矩阵，并将这个调整后的数据传入sess.run过程
            #reshaped_xs = np.reshape(xs_reshape,(-1,mnist_inference_LeNet5.IMAGE_SIZE,mnist_inference_LeNet5.IMAGE_SIZE,mnist_inference_LeNet5.NUM_CHANNELS))
            _,step = sess.run([train_op,global_step],feed_dict={x:xs_reshape,y_:ys})
            if i % 50 == 0:
                loss_value,summary = sess.run([loss,merged],feed_dict={x:xs_reshape,y_:ys})
                test_accuracy = sess.run(accuracy,feed_dict=test_feed)
                writer.add_summary(summary,i)
                plot_save = (i, loss_value, test_accuracy)
                plot_saves.append(plot_save)
                print("在第%d轮训练后，训练集的误差为: %g,测试集的准确率为：%g"%(i,loss_value,test_accuracy))
        print(("After %d training step(s), test accuracy using average model is %g" % (i, test_accuracy)))
    return plot_saves

def plot_accuracy(plot_saves):
    m = len(plot_saves)
    x_epoach = []#保存训练次数的列表
    y_train_error = []#保存训练集误差的列表
    test_acc = []#保持测试集准确率列表
    for i in range(m):
        x_epoach.append(plot_saves[i][0])
        y_train_error.append(plot_saves[i][1])
        test_acc.append(plot_saves[i][2])
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    lns1 = ax1.plot(x_epoach,y_train_error,'m',label = 'train error')
    ax2 = ax1.twinx()
    lns2 = ax2.plot(x_epoach, test_acc, 'r', label='test accuracy')
    #合并图例
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=1, frameon=False)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('train_error')
    ax1.set_title('MNIST训练集、测试集', fontproperties=font)  # 绘制图形的标题
    ax2.set_ylabel("test_acc")
    plt.show()

def main():
    mnist = input_data.read_data_sets(r"MNIST_data",one_hot=True)
    plot_saves = train(mnist)
    plot_accuracy(plot_saves)

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("总花费时间为："+str((end-start)/60)+'min')

