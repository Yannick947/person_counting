import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(model, datagen_test):
    datagen_test.reset_label_states()

    predictions = model.predict_generator(generator=datagen_test.datagen(), steps=10)
    predictions_inverse = datagen_test.scaler.scaler_labels.inverse_transform(predictions)

    y_true = datagen_test.scaler.scaler_labels.inverse_transform(datagen_test.get_labels())

    plt.scatter(predictions_inverse, y_true)
    plt.title('Predictions over ground truth')
    plt.xlabel('Predictions')
    plt.ylabel('Ground truth')

    plt.xticks(np.arange(min(np.append(predictions_inverse, y_true)),
                         max(np.append(predictions_inverse , y_true))))

    plt.yticks(np.arange(min(np.append(predictions_inverse, y_true)),
                         max(np.append(predictions_inverse , y_true))))
    plt.show()


def plot(history):
    # plot history
    plt.plot(history.history['loss'], label='train')
    print(history.history.keys())
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()