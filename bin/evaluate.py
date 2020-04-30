import math

from keras.models import load_model

from person_counting.utils.visualization_utils import plot, visualize_predictions

def evaluate_model(model, history, gen, mode, visualize=True):

    if type(model) == list:
        model = load_model(model) 
    
    pred_test, y_true_test = get_predictions(model, gen)
    evaluate_predictions(history, pred_test, y_true_test, visualize=True, mode=mode)


def evaluate_predictions(history, predictions, y_true, visualize, mode, ipython_mode=True):
    difference = 0
    for prediction, y in zip(predictions, y_true):
        difference += abs(prediction - y)
    
    mean_difference = difference / len(y_true)
    print('For mode: ', mode)
    print('Mean difference between Ground truth and predictions is: ', mean_difference)

    if visualize == True: 
        plot(history, ipython_mode) 
        visualize_predictions(predictions, y_true, ipython_mode=ipython_mode)


def get_predictions(model, gen): 

    gen.reset_label_states()

    predictions = model.predict_generator(generator=gen.datagen(), steps=int(len(gen)))
    if gen.scaler is not None: 
        predictions_inverse = gen.scaler.scaler_labels.inverse_transform(predictions)
        y_true = gen.scaler.scaler_labels.inverse_transform(gen.get_labels())
        return predictions_inverse, y_true

    else: 
        y_true = gen.get_labels()
        print('No scaler found, calculations done on data given')
        return predictions, y_true
    








