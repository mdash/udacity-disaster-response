# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:5000/

### Description
The app is a classifier to categorize messages during natural disasters based on the type of aid required to help prioritize and deploy resources based on needs.

### Model details
A Support Vector Classifier was used on the training data. As is evident from the second visualization in the app (snapshot below for reference), the distribution of messages into classes is highly imbalanced. This was handled by weighting samples in the positive class (using sklearn package)

![distribution of message categories][message_category_graph]

[message_category_graph]: ./message_categories.png "Logo Title Text 2"
