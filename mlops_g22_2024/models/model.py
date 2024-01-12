
from transformers import AutoModelForSequenceClassification
​
# Defining dictionaries to map between the numerical ID's and the corresponding emotion labels.
# This helps in interpreting the model's output more intuitively.
id2label = {0:'anger', 1:'fear', 2:'joy', 3:'love', 4:'sadness', 5:'surprise'}
label2id = {'anger':0, 'fear':1, 'joy':2, 'love':3, 'sadness':4, 'surprise':5}
​
'''
 Initializing the model for sequence classification. 
 The model is configured to classify into 6 different labels as defined above.
'''
​
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=6, id2label=id2label, label2id=label2id)