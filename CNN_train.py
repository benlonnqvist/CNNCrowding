import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from densenet121 import DenseNet


def main():

    batch_size = 8

    # load dataset
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory(r'./dir/train', target_size=(224, 224),
                                                  batch_size=batch_size, class_mode='categorical',
                                                  shuffle=True)
    test_generator = datagen.flow_from_directory(r'./dir/test', target_size=(224, 224),
                                                  batch_size=batch_size, class_mode='categorical',
                                                  shuffle=True)
    val_generator = datagen.flow_from_directory(r'./dir/val', target_size=(224, 224),
                                                  batch_size=batch_size, class_mode='categorical',
                                                  shuffle=True)

    save_path = 'savepath.h5'
    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    checkpoint_list = [checkpoint]
    model = DenseNet(reduction=0.5, weights_path=None)
    adam = keras.optimizers.Adam(lr=0.001)
    print(model.summary())

    # Freeze layers if desired
    # for layer in model.layers[:len(model.layers)-10]:
    #     layer.trainable = False

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(r'./weights.h5')
    model.fit_generator(train_generator, steps_per_epoch=int(26400/batch_size), epochs=100, verbose=1,
                        validation_data=val_generator, validation_steps=int(3316/batch_size), callbacks=checkpoint_list)


if __name__ == '__main__':

    main()
