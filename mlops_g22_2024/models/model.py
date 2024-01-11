from transformers import DistilBertForSequenceClassification
from transformers import AutoModelForSequenceClassification

'''# Load pre-trained model with a classification head
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=6,
    output_attentions=False,
    output_hidden_states=False,
)'''
# Defining labels and corresponding ids
id2label = {0:'anger', 1:'fear', 2:'joy', 3:'love', 4:'sadness', 5:'surprise'}
label2id = {'anger':0, 'fear':1, 'joy':2, 'love':3, 'sadness':4, 'surprise':5}

# Defining the model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=6, id2label=id2label, label2id=label2id
)
