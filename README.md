# Email Author Identification

This project is a project that attempts to identify authors of emails from the enron dataset. It does so using nlp and machine learning techniques.

## Instructions For Use

Required python modules can be found in `requirements.lock`.

1. Run `python data_cleaner.py`. This gets the enron email dataset and stores a cleaned version of it in `./data/emails.csv`.
2. Run `python email_tokenizer.py`. This reads the cleaned email and tokenizes the contents of all the emails using term frequency - inverse document frequency encoding. Tokenized emails are stored in `./data/tf-idf.csv`. _This will take a while (1 hour maybe)_

Work in Progress
