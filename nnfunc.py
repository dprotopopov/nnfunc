import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x0, x1 = 10, 20 # диапазон аргумента функции

test_data_size = 2000 # количество данных для итерации обучения
iterations = 20000 # количество итераций обучения
learn_rate = 0.01 # коэффициент переобучения

hiddenSize = 10 # размер скрытого слоя

# функция генерации тестовых величин
def generate_test_values():
    train_x = []
    train_y = []

    for _ in range(test_data_size):
        x = x0+(x1-x0)*np.random.rand()
        y = math.sin(x) # исследуемая функция
        train_x.append([x])
        train_y.append([y])

    return np.array(train_x), np.array(train_y)


# узел на который будем подавать аргументы функции
x = tf.placeholder(tf.float32, [None, 1], name="x")

# узел на который будем подавать значения функции
y = tf.placeholder(tf.float32, [None, 1], name="y")

# скрытый слой
nn = tf.layers.dense(x, hiddenSize,
                     activation=tf.nn.sigmoid,
                     kernel_initializer=tf.initializers.ones(),
                     bias_initializer=tf.initializers.random_uniform(minval=-x1, maxval=-x0),
                     name="hidden")

# выходной слой
model = tf.layers.dense(nn, 1,
                        activation=None,
                        name="output")

# функция подсчёта ошибки
cost = tf.losses.mean_squared_error(y, model)

train = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

init = tf.initializers.global_variables()

with tf.Session() as session:
    session.run(init)

    for _ in range(iterations):

        train_dataset, train_values = generate_test_values()

        session.run(train, feed_dict={
            x: train_dataset,
            y: train_values
        })

        if(_ % 1000 == 999):
            print("cost = {}".format(session.run(cost, feed_dict={
                x: train_dataset,
                y: train_values
            })))

    train_dataset, train_values = generate_test_values()

    train_values1 = session.run(model, feed_dict={
        x: train_dataset,
    })

    plt.plot(train_dataset, train_values, "bo",
             train_dataset, train_values1, "ro")
    plt.show()

    with tf.variable_scope("hidden", reuse=True):
        w = tf.get_variable("kernel")
        b = tf.get_variable("bias")
        print("hidden:")
        print("kernel=", w.eval())
        print("bias = ", b.eval())
    
    with tf.variable_scope("output", reuse=True):
        w = tf.get_variable("kernel")
        b = tf.get_variable("bias")
        print("output:")
        print("kernel=", w.eval())
        print("bias = ", b.eval())
