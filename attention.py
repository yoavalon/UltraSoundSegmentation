import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io


def cnn(input, reuse = tf.AUTO_REUSE) :
    if (reuse):
        tf.get_variable_scope().reuse_variables()         
    
    with tf.variable_scope("action.CNN/CNN", reuse=tf.AUTO_REUSE) :
        with tf.variable_scope("conv1") as scope:
            net = tf.layers.conv2d(input,filters=16, kernel_size=[4, 4], padding="same", strides=(2, 2), activation=tf.nn.relu, reuse= tf.AUTO_REUSE)            
        with tf.variable_scope("conv2") as scope:
            net = tf.layers.conv2d(net,filters=16, kernel_size=[3, 3], padding="same", strides=(2, 2), activation=tf.nn.relu, reuse= tf.AUTO_REUSE)                    
        with tf.variable_scope("conv3") as scope:
             net = tf.layers.conv2d(net,filters=16, kernel_size=[2, 2], padding="same", strides=(2, 2), activation=tf.nn.relu, reuse= tf.AUTO_REUSE)                                
        with tf.variable_scope("conv4") as scope:
             net = tf.layers.conv2d(net,filters=32, kernel_size=[2, 2], padding="same", strides=(2, 2), activation=tf.nn.relu, reuse= tf.AUTO_REUSE)

        with tf.variable_scope("dense") as scope:              
            net = tf.contrib.layers.flatten(net)                       
            net = tf.layers.dense(inputs=net, units=1080, activation=None, reuse = tf.AUTO_REUSE)  
            
    return net

def fill_contours(arr):
    return np.maximum.accumulate(arr,1) & \
           np.maximum.accumulate(arr[:,::-1],1)[:,::-1]

def loadLabel(index) : 
    img = Image.open(f'./training_set/{index:03}_HC_Annotation.png')
    img = np.asarray(img)
    img = fill_contours(img)

    im = Image.fromarray(img)
    im = im.resize(size=(40,27))
    img = (np.asarray(im)/255).flatten()

    return img

def CreateBatch(batchSize) : 

    indexes = np.random.randint(804, size=batchSize)

    images = np.asarray([np.asarray(Image.open(f'./training_set/{i:03}_HC.png').resize(size=(200,135)), dtype="int32") for i in indexes])
    labels = np.asarray([loadLabel(i) for i in indexes])

    return images, labels


input = tf.placeholder(shape = (None,135,200,1), dtype = tf.float32)
labels = tf.placeholder(shape = (None, 1080), dtype = tf.float32)

logits = cnn(input)
pred = logits[0]

with tf.variable_scope('action.LOSS', reuse=tf.AUTO_REUSE) :
    crossEntrop = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels = labels)
    loss = tf.reduce_sum(crossEntrop)    
    optimizer = tf.train.AdamOptimizer(0.0001, name = "action.AdamOptimizer")    
    train_op = optimizer.minimize(loss=loss)    

with tf.Session() as sess:
    
    writer = tf.summary.FileWriter('./log', sess.graph)
    sess.run(tf.global_variables_initializer())     # Initialize all tensorflow variables
    sess.run(tf.local_variables_initializer())

    for episode in range(50000) :        

        images, labs = CreateBatch(20)       
        images = np.expand_dims(images, 3)
        
        for step in range(10) :                 #100

            _, los, logs, pre  = sess.run([
                train_op, 
                loss, 
                logits, 
                pred
            ] ,feed_dict = {input: images, labels: labs})

        print(f'{episode} {los:10.6f}')

        summary=tf.Summary()
        summary.value.add(tag='Loss', simple_value = los)  # reward per episode                        
        writer.add_summary(summary, episode)
        
        if (episode % 2==0) and (episode>0) :

            preImg = pre.reshape((27,40))
            original = np.squeeze(images[0])
            pic = Image.fromarray(preImg)
            pic = pic.resize(size=(200,135))


            plt.imshow(np.squeeze(images[0]), cmap='bone')
            plt.imshow(pic,cmap='Blues', alpha=0.6)

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=200)
            buf.seek(0)                        
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)            
            
            sum1 = tf.summary.image(f'{episode}/overlay', image)
            writer.add_summary(sum1.eval(session = sess), episode)       
            
            plt.close()

            
            
            plt.imshow(np.squeeze(images[0]), cmap='bone')            

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=200)
            buf.seek(0)                        
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)            
            
            sum1 = tf.summary.image(f'{episode}/original', image)
            writer.add_summary(sum1.eval(session = sess), episode)       
            
            plt.close()


            
            plt.imshow(pic, cmap='GnBu')

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)                        
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)            
            
            sum1 = tf.summary.image(f'{episode}/prediction', image)
            writer.add_summary(sum1.eval(session = sess), episode)       
            
            plt.close()


            ll = np.squeeze(labs[0]).reshape((27,40))
            plt.imshow(ll, cmap = 'bone')

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)                        
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)            
            
            sum1 = tf.summary.image(f'{episode}/groundtruth', image)
            writer.add_summary(sum1.eval(session = sess), episode)       
            
            plt.close()
