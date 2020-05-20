### twitter_disaster_classification
A web app predicting whether a tweet refers to a disaster or a call for help.

##### Python Libraries

nltk (3.2.5)
scikit-learn (0.19.1)
numpy (1.12.1)  
pandas (0.23.3)
matplotlib (2.1.0)
Flask (0.12.4)

##### Files
/data/**disaster_messages.csv** - messages/tweets in their original language, and english translation with id numbers
/data/**disaster_categories.csv** - categorized message ids
**process_data.py** - processing script that transforms the data (disaster_categories.csv, disaster_messages.csv) into a usable format for the machine learning step

/models/**train_classifier.py** - trains a model to classifiy messages, and also outputs performance metrics of the model

/app/**master.html** - the html front end for the web app to be used
/app/**templates/go.html** - enhances the master.html
/app/**run.py** - python script to run the webapp

##### Build

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
