import boto3
import email
import json
import urllib.parse
import string
import sys
import numpy as np
import json

from hashlib import md5
import boto3

sage = boto3.client('sagemaker-runtime')
sqs = boto3.resource('sqs')
s3 = boto3.client('s3')
bucket = 'aatman.ml-emails'

vocabulary_length = 9013
endpoint_name = "sms-spam-classifier-mxnet-2022-05-05-16-22-56-540" 


if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans
    
def vectorize_sequences(sequences, vocabulary_length):
    results = np.zeros((len(sequences), vocabulary_length))
    for i, sequence in enumerate(sequences):
       results[i, sequence] = 1. 
    return results

def one_hot_encode(messages, vocabulary_length):
    data = []
    for msg in messages:
        temp = one_hot(msg, vocabulary_length)
        data.append(temp)
    return data

def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    """Converts a text to a sequence of words (or tokens).
    # Arguments
        text: Input text (string).
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to convert the input to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of words (or tokens).
    """
    if lower:
        text = text.lower()

    if sys.version_info < (3,):
        if isinstance(text, unicode):
            translate_map = dict((ord(c), unicode(split)) for c in filters)
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = dict((c, split) for c in filters)
        translate_map = maketrans(translate_dict)
        text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]

def one_hot(text, n,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '):
    """One-hot encodes a text into a list of word indexes of size n.
    This is a wrapper to the `hashing_trick` function using `hash` as the
    hashing function; unicity of word to index mapping non-guaranteed.
    # Arguments
        text: Input text (string).
        n: int. Size of vocabulary.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        List of integers in [1, n]. Each integer encodes a word
        (unicity non-guaranteed).
    """
    return hashing_trick(text, n,
                         hash_function='md5',
                         filters=filters,
                         lower=lower,
                         split=split)


def hashing_trick(text, n,
                  hash_function=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=' '):
    """Converts a text to a sequence of indexes in a fixed-size hashing space.
    # Arguments
        text: Input text (string).
        n: Dimension of the hashing space.
        hash_function: defaults to python `hash` function, can be 'md5' or
            any function that takes in input a string and returns a int.
            Note that 'hash' is not a stable hashing function, so
            it is not consistent across different runs, while 'md5'
            is a stable hashing function.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of integer word indices (unicity non-guaranteed).
    `0` is a reserved index that won't be assigned to any word.
    Two or more words may be assigned to the same index, due to possible
    collisions by the hashing function.
    The [probability](
        https://en.wikipedia.org/wiki/Birthday_problem#Probability_table)
    of a collision is in relation to the dimension of the hashing space and
    the number of distinct objects.
    """
    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        hash_function = lambda w: int(md5(w.encode()).hexdigest(), 16)

    seq = text_to_word_sequence(text,
                                filters=filters,
                                lower=lower,
                                split=split)
    return [int(hash_function(w) % (n - 1) + 1) for w in seq]


def get_prediction(blob):
    # test_messages = ["What time are you free tomorrow?"]
    test_messages = [blob]
    one_hot_test_messages = one_hot_encode(test_messages, vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
    
    response = sage.invoke_endpoint(
        EndpointName=endpoint_name, 
        Body=json.dumps(encoded_test_messages.tolist()), 
        ContentType='application/json'
    )
    
    response_body = response['Body'].read().decode('utf-8')
    response_body_json = json.loads(response_body)
    predicted_label_val = response_body_json["predicted_label"][0][0]
    predicted_probability = response_body_json["predicted_probability"][0][0]
    if int(predicted_label_val) == 0:
        predicted_label = "Ham"
    else:
        predicted_label = "Spam"
    return (predicted_label, predicted_probability)

def send_email(predicted_label, predicted_probability):
    email = ''
    email_receive_date = ''
    email_subject = ''
    email_body = ''
    classification = predicted_label
    classification_confidence_score = predicted_probability
    body_length = len(email_body)
    if body_length > 240:
        body_length = 240
    new_body = "We received your email sent at " + email_receive_date 
    + " with the subject " + email_subject 
    + ".\nHere is a 240 character sample of the email body:\n" 
    + email_body[:body_length] + "\nThe email was categorized as " 
    + classification + " with a " + classification_confidence_score 
    + "% confidence.\n"
    
    response = mail.send_email(
        Source='am11444@nyu.edu',
        Destination={'ToAddresses': [email]},
        Message={
            'Subject': {
                'Data': 'Spam Prediction'
            },
            'Body': {
                'Text': {
                    'Data': new_body
                }
            }
        }
    )
    return

def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        raw_email = response['Body'].read().decode('utf-8')
        # print("lines:", lines)
        my_mail = email.message_from_string(raw_email)
        blob = ""
        if my_mail.is_multipart():
            lines = my_mail.get_payload()[0].get_payload().strip('\n').split('\n')
            for line in lines:
                line = line.strip('\r')
                blob += line + '. '
        else:
            print("SPAMMER -- ERROR: SINGLEPART MAIL")
            print(b.get_payload())
        print("blob:")
        blob = blob[:len(blob)-1]
        print(blob)
        # get_prediction(blob)
        predicted_label, predicted_probability = get_prediction(blob)
        print("predicted_label: ", predicted_label)
        print("predicted_probability: ", predicted_probability)
        send_email(predicted_label, predicted_probability)
        return
    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
        raise e
