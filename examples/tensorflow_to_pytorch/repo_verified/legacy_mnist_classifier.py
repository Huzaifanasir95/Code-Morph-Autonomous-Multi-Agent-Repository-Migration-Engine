"""
Legacy TensorFlow 1.x Image Classifier

This is an example of legacy code using deprecated TensorFlow 1.x APIs.
This code demonstrates common patterns that need migration to modern frameworks.
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def create_model(input_shape, num_classes):
    """
    Create a simple neural network model using TensorFlow 1.x APIs

    Args:
        input_shape: Shape of input data
        num_classes: Number of output classes

    Returns:
        Model logits
    """
    # Use deprecated placeholder API
    x = tf.placeholder(tf.float32, shape=[None, input_shape], name="input")
    y = tf.placeholder(tf.float32, shape=[None, num_classes], name="labels")

    # Define model architecture
    with tf.variable_scope("layer1"):
        W1 = tf.get_variable("weights", [input_shape, 256], initializer=tf.random_normal_initializer())
        b1 = tf.get_variable("bias", [256], initializer=tf.zeros_initializer())
        hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    with tf.variable_scope("layer2"):
        W2 = tf.get_variable("weights", [256, 128], initializer=tf.random_normal_initializer())
        b2 = tf.get_variable("bias", [128], initializer=tf.zeros_initializer())
        hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)

    with tf.variable_scope("output"):
        W3 = tf.get_variable("weights", [128, num_classes], initializer=tf.random_normal_initializer())
        b3 = tf.get_variable("bias", [num_classes], initializer=tf.zeros_initializer())
        logits = tf.matmul(hidden2, W3) + b3

    return x, y, logits


def train_model(data_dir="./data"):
    """
    Train the model using deprecated TensorFlow 1.x Session API

    Args:
        data_dir: Directory containing training data
    """
    # Load data using deprecated API
    mnist = input_data.read_data_sets(data_dir, one_hot=True)

    # Model parameters
    input_shape = 784
    num_classes = 10
    learning_rate = 0.001
    batch_size = 128
    num_epochs = 10

    # Create model
    x, y, logits = create_model(input_shape, num_classes)

    # Define loss and optimizer using TF1 APIs
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    )

    # Use deprecated optimizer API
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy)

    # Accuracy metric
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Use deprecated Session API
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)

        print("Starting training...")

        # Training loop
        for epoch in range(num_epochs):
            total_batches = mnist.train.num_examples // batch_size

            for batch in range(total_batches):
                batch_x, batch_y = mnist.train.next_batch(batch_size)

                # Run training step
                _, loss, acc = sess.run(
                    [train_op, cross_entropy, accuracy],
                    feed_dict={x: batch_x, y: batch_y}
                )

                if batch % 100 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

        # Evaluate on test set
        test_accuracy = sess.run(
            accuracy,
            feed_dict={x: mnist.test.images, y: mnist.test.labels}
        )
        print(f"\nTest Accuracy: {test_accuracy:.4f}")

        # Save model using deprecated Saver API
        saver = tf.train.Saver()
        save_path = saver.save(sess, "./models/mnist_model.ckpt")
        print(f"Model saved to: {save_path}")


def evaluate_model(model_path, test_data):
    """
    Evaluate saved model using Session API

    Args:
        model_path: Path to saved model checkpoint
        test_data: Test dataset
    """
    # Restore model
    with tf.Session() as sess:
        # Load model using deprecated API
        saver = tf.train.import_meta_graph(model_path + ".meta")
        saver.restore(sess, model_path)

        # Get tensors by name
        x = sess.graph.get_tensor_by_name("input:0")
        logits = sess.graph.get_tensor_by_name("output/add:0")

        # Run prediction
        predictions = sess.run(logits, feed_dict={x: test_data})

    return predictions


if __name__ == "__main__":
    # Train the model
    train_model()

    print("\n✅ Training complete!")
    print("⚠️  This code uses deprecated TensorFlow 1.x APIs")
    print("📋 Use Code-Morph to migrate to modern frameworks!")
