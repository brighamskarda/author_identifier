# Copyright (c) 2025 Brigham Skarda

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.text import TextCollection
from nltk.tokenize import word_tokenize
from collections.abc import Iterable


def main():
    print("Reading emails.csv")
    email_data = pd.read_csv("./data/emails.csv")

    print("Fitting tokenizer")
    tfidf = TfidfVectorizer(
        tokenizer=word_tokenize, preprocessor=None, token_pattern=None
    )
    tfidf.fit(email_data["content"][:1000])

    print("Getting email tokenizations")
    tokenized_emails = tfidf.transform(email_data["content"][:1000])

    print("Reading into dataframe for export")
    df = pd.DataFrame(tokenized_emails.toarray(), columns=tfidf.get_feature_names_out())

    print("Exporting to csv")
    df.to_csv("./data/tf-idf.csv")


# def get_tokenized_emails(
#     emails: pd.DataFrame, tokenizer: EmailTokenizer, length=0
# ) -> pd.DataFrame:
#     """
#     Converts a dataframe of emails to two dataframes of tokenized emails and authors.

#     Args:
#         emails (pd.DataFrame): Should contains columns 'content' and 'author'.
#         tokenizer (EmailTokenizer):
#         length (int, optional): The number of tokens to generate per email. If an email contains more tokens than length then tokens after length are not included. If an email has fewer tokens then extra tokens are set to 0. If length is set to 0, then length is set so that all emails contain all their tokens. Defaults to 0.

#     Returns:
#         tuple[pd.DataFrame, pd.DataFrame]: The first value is the tokenized email with columns 'token_1'...'token_length'.
#         The second value is a one hot encoding of the author.
#     """
#     tokenized_contents: list[list[float]] = []
#     for i, email_content in enumerate(emails["content"]):
#         print(f"Tokenizing emails {i}/{len(emails['content'])}")
#         tokenized_contents.append(tokenizer.tf_idf(email_content))
#     if length == 0:
#         length = get_max_token_length(tokenized_contents)
#     pad_truncate_embeddings(tokenized_contents, length)
#     return pd.DataFrame(tokenized_contents)


# def get_max_token_length(tokenized_emails: list[list[float]]) -> int:
#     max_len = 0
#     for email in tokenized_emails:
#         if len(email) > max_len:
#             max_len = len(email)
#     return max_len


# def pad_truncate_embeddings(tokenized_contents: list[list[float]], length: int):
#     """Pads or truncates tokenized emails depending on length"""
#     for email in tokenized_contents:
#         while len(email) < length:
#             email.append(0.0)
#         while len(email) > length:
#             email.pop()


if __name__ == "__main__":
    main()
