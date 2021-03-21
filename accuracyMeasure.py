from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from time import time
import joblib


# def nfeature_accuracy_checker(st,vectorizer=None, n_features=None, stop_words=None, ngram_range=(1, 1), classifier=None,x_train=None, y_train=None, x_test=None, y_test=None):
#     result = []
#     st.write("Using Classifier ",classifier)
#     vectorizer.set_params(stop_words=stop_words, max_features=n_features , ngram_range = ngram_range)
#     checker_pipeline = Pipeline([
#             ('vectorizer', vectorizer),
#             ('classifier', classifier)
#     ])
#     st.write(f'Validation result for {n_features}')
#     nfeature_accuracy, tt_time, sentiment_fit = accuracy_summary(st, checker_pipeline, x_train, y_train, x_test, y_test)
#     result.append((nfeature_accuracy, tt_time))
#     return sentiment_fit


# def accuracy_summary(st, pipeline, x_train, y_train, x_test, y_test):
#     t0 = time()
#     st.write("Doing training ", x_train.shape, " y_train is ", y_train.shape)
#     sentiment_fit = pipeline.fit(x_train, y_train)
#     y_pred = sentiment_fit.predict(x_test)
#     test_train_time = time() - t0
#     joblib.dump(sentiment_fit,'trainedModel.pkl')
#     accuracy = accuracy_score(y_test, y_pred)
#     st.write(f"Accuracy score : {accuracy*100} ")
#     st.write(f"Train and Test time: {test_train_time}")
#     return accuracy, test_train_time,sentiment_fit

def run_classification(st,sent140_train_X, sent140_train_Y, sent140_dev_X, sent140_dev_Y,model):
  st.write("\n\nModel Is ",model)
  model.fit(sent140_train_X,sent140_train_Y)
  sent140_train_preds = model.predict(sent140_train_X)
  sent140_dev_preds = model.predict(sent140_dev_X)
  
  # print(emoji_train_preds,emoji_dev_preds)

  st.write("\n\n**** ACCURACY OF MODEL ON TRAINING DATASET DATASET **** \n\n")
  
  
  nr_correct = (sent140_train_Y== model.predict(sent140_train_X)).sum()
  nr_incorrect = (sent140_train_Y!= model.predict(sent140_train_X)).sum()
  st.write(f'Correctly classified : {nr_correct} \nInCorrectly classified : {nr_incorrect}')
  fraction_wrong = nr_incorrect / (nr_correct+nr_incorrect)
  fraction_right = nr_correct / (nr_correct+nr_incorrect)
  st.write(f'Correctly classified : {fraction_right*100} \nInCorrectly classified : {fraction_wrong*100}')
  
  
  st.write("\n\n**** ACCURACY OF MODEL ON TESTING DATASET  **** \n\n")

  nr_correct = (sent140_dev_Y== model.predict(sent140_dev_X)).sum()
  nr_incorrect = (sent140_dev_Y!= model.predict(sent140_dev_X)).sum()
  st.write(f'Correctly classified : {nr_correct} \nInCorrectly classified : {nr_incorrect}')

  fraction_wrong = nr_incorrect / (nr_correct+nr_incorrect)
  fraction_right = nr_correct / (nr_correct+nr_incorrect)
  st.write(f'Correctly classified : {fraction_right*100} \nInCorrectly classified : {fraction_wrong*100}')
  
  



  st.write("\n\n**** METRICS OF MODEL ON TRAINING DATASET  ****\n\n")
  st.write(classification_report(sent140_train_Y,sent140_train_preds))
  st.write(confusion_matrix(sent140_train_Y, sent140_train_preds))

  
  st.write("\n\n**** METRICS OF MODEL ON TESTING DATASET ****\n\n")
  st.write(classification_report(sent140_dev_Y,sent140_dev_preds))
  st.write(confusion_matrix(sent140_dev_Y, sent140_dev_preds))

  
