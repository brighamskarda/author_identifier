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

from nltk.text import TextCollection
from nltk.tokenize import word_tokenize
from collections.abc import Iterable


class EmailTokenizer:
    """
    Takes in an entire email corpus. It can then provide term frequency - inverse document frequency (tf-idf) encodings for specific emails.

    The corpus or emails provided for tokenization need not be cleaned.
    """

    def __init__(self, corpus: Iterable[str]):
        """
        Args:
            corpus (Iterable[str]): A list of email bodies.
        """
        tokenized_emails: list[list[str]] = []
        for email in corpus:
            tokenized_emails.append(EmailTokenizer.__pre_tokenize(email))
        self.__collection = TextCollection(tokenized_emails)

    def tf_idf(self, email: str) -> list[float]:
        """Provides term frequency - inverse document frequency encoding for a specific email in the context the of the whole corpus."""
        tokens = EmailTokenizer.__pre_tokenize(email)
        tf_idf_tokens = []
        for word in tokens:
            tf_idf_tokens.append(self.__collection.tf_idf(word, tokens))
        return tf_idf_tokens

    def __pre_tokenize(text: str) -> list[str]:
        """
        Splits a string into tokens. Tokens are usually representative of a word, but can also be punctuation or lemmas.

        The description of this function is intentionally vague so that pre-tokenization (before tf_idf) can be flexible while still being the same for both __init__() and tf_idf()
        """
        return word_tokenize(text)


def main():
    print("Reading emails.csv")
    email_data = pd.read_csv("./data/emails.csv")

    print("Building email tokenizer")
    tokenizer = EmailTokenizer(email_data["content"])

    print("Getting email tokenizations")
    x = get_tokenized_emails(email_data, tokenizer)
    x.to_csv("./data/tf-idf.csv")


def get_tokenized_emails(
    emails: pd.DataFrame, tokenizer: EmailTokenizer, length=0
) -> pd.DataFrame:
    """
    Converts a dataframe of emails to two dataframes of tokenized emails and authors.

    Args:
        emails (pd.DataFrame): Should contains columns 'content' and 'author'.
        tokenizer (EmailTokenizer):
        length (int, optional): The number of tokens to generate per email. If an email contains more tokens than length then tokens after length are not included. If an email has fewer tokens then extra tokens are set to 0. If length is set to 0, then length is set so that all emails contain all their tokens. Defaults to 0.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The first value is the tokenized email with columns 'token_1'...'token_length'.
        The second value is a one hot encoding of the author.
    """
    tokenized_contents: list[list[float]] = []
    for i, email_content in enumerate(emails["content"]):
        if i % (len(emails["content"]) / 1000) == 0:
            print(f"Tokenizing emails {i}/{len(emails['content'])}")
        tokenized_contents.append(tokenizer.tf_idf(email_content))
    if length == 0:
        length = get_max_token_length(tokenized_contents)
    pad_truncate_embeddings(tokenized_contents, length)
    return pd.DataFrame(tokenized_contents)


def get_max_token_length(tokenized_emails: list[list[float]]) -> int:
    max_len = 0
    for email in tokenized_emails:
        if len(email) > max_len:
            max_len = len(email)
    return max_len


def pad_truncate_embeddings(tokenized_contents: list[list[float]], length: int):
    """Pads or truncates tokenized emails depending on length"""
    for email in tokenized_contents:
        while len(email) < length:
            email.append(0.0)
        while len(email) > length:
            email.pop()


if __name__ == "__main__":
    main()
