import tensorflow as tf
import pathlib

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


def print_version():
    print(tf.__version__)


def import_data_and_teach():
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        'photos/training/',
        validation_split=0.1,
        subset="training",
        seed=123,
        image_size=(100, 100)
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        'photos/training/',
        validation_split=0.1,
        subset="validation",
        seed=123,
        image_size=(100, 100)
    )

    class_names = train_dataset.class_names
    print(class_names)
    for images, labels in train_dataset:
        print(images, labels)

    test_dataset = tf.keras.utils.image_dataset_from_directory('photos/training/')
    print(test_dataset.class_names)

    # Создаем последовательную модель
    model = tf.keras.Sequential()
    # Сверточный слой
    model.add(tf.keras.layers.Conv2D(16, (5, 5), padding='same',
                                     input_shape=(100, 100, 3), activation='relu'))
    # Слой подвыборки
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # Сверточный слой
    model.add(tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same'))
    # Слой подвыборки
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # Сверточный слой
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
    # Слой подвыборки
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # Сверточный слой
    model.add(tf.keras.layers.Conv2D(128, (5, 5), activation='relu', padding='same'))
    # Слой подвыборки
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # Полносвязная часть нейронной сети для классификации
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    # Выходной слой, 2 нейрона по количеству классов
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])

    history = model.fit(train_dataset,
                        validation_data=validation_dataset,
                        epochs=10,
                        verbose=2)

    test_acc = model.evaluate(validation_dataset, verbose=2)
    print('\nTest accuracy:', test_acc)

    # probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    # predictions = probability_model.predict(train_dataset(2))
    # print(predictions)

    # img = tf.keras.preprocessing.image.load_img('photos/training/me/me004.jpg', target_size=(100, 100), color_mode="grayscale")
    # plt.imshow(img, cmap='gray')
    # plt.show()
    # # Преобразуем картинку в массив
    # x = tf.keras.preprocessing.image.img_to_array(img)
    # # Меняем форму массива в плоский вектор
    # # x = x.reshape(1, 10000)
    # # Инвертируем изображение
    # x = 255 - x
    # # Нормализуем изображение
    # x /= 255
    # np.expand_dims(x, axis=0)
    #
    # # probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    # # prediction = probability_model.predict(x)
    prediction = model.predict(validation_dataset)
    print(prediction)


if __name__ == '__main__':
    print_version()
    import_data_and_teach()
