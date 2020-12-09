import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
import pickle
import pandas as pd


def score_model(self):
    print("Model Scoring")
    # Create spark session
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    
    # Unpickle, pkl file
    model_rdd_pkl = sc.binaryFiles("model.pkl")
    model_rdd_data = model_rdd_pkl.collect()
    
    # Load and broadcast python object over spark nodes
    creditcardfrauddetection_model = pickle.loads(model_rdd_data[0][1])
    broadcast_creditcardfrauddetection_model = sc.broadcast(creditcardfrauddetection_model)
    print(broadcast_creditcardfrauddetection_model.value)
    
    # Load dataset for prediction
    dataset = pd.read_csv('creditcard.csv')
    dataset.head()
    X = dataset.drop(["Class"], axis = 1)
    y = dataset["Class"]
    X1 = X.drop(["Time"], axis = 1)
    
    # Create feature column list on which model was trained
    feature_columns = X1.columns.to_list()
    print(feature_columns)
    
    # Create spark dataframe for prediction
    df = spark.read.csv('creditcard.csv', header=True)
    df.show()
    
    # Create udf and call predict method on broadcasted model
    def predict(*cols):
    prediction = broadcast_creditcardfrauddetection_model.value.predict_proba((cols,))
    return float(prediction[0,1])

    predict_udf = udf(predict, DoubleType())
    
    # Create predictions using spark udf
    df = df.withColumn("score", predict_udf(*feature_columns))
    df.show()

    # Calculate accuracy
    count = df.count()
    correct_predictions = df.filter(col("Class")==col("score")).count()
    incorrect_predictions = df.filter(col("Class")!=col("score")).count()
    accuracy = (correct_predictions/count)*100
    print(accuracy)
    

if __name__ == "__main__":
    score_model()
