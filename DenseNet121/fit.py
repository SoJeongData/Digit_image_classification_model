model.fit(train_generator,
          epochs=20,
          validation_data=test_generator,
          verbose=0,
          callbacks=[TqdmProgressCallback(), cp_cb, es_cb]
          )

train_loss = total_history['loss']
train_acc = total_history['acc']

valid_loss = total_history['val_loss']
valid_acc = total_history['val_acc']

plt.plot(train_loss,
         'o',
         color='r',
         label='training loss')
plt.plot(valid_loss,
         color='b',
         label='validation loss')
plt.legend()
plt.show()

plt.plot(train_acc,
         'o',
         color='r',
         label='training accuracy')
plt.plot(valid_acc,
         color='b',
         label='validation accuracy')
plt.legend()
plt.show()

model.save('/content/drive/MyDrive/save-model/02.21_model.h5')
