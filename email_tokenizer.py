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

from nltk.text import TextCollection
from nltk.tokenize import word_tokenize


class EmailTokenizer:
    """
    Takes in an entire email corpus. It can then provide term frequency - inverse document frequency (tf-idf) encodings for specific emails.

    The corpus or emails provided for tokenization need not be cleaned.
    """

    def __init__(self, corpus: list[str]):
        """
        Args:
            corpus (list[str]): A list of email bodies.
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
