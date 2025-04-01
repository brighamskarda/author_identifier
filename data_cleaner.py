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

# This is a python script designed to clean and prepare the enron email dataset from https://www.kaggle.com/datasets/wcukierski/enron-email-dataset. It automatically downloads the dataset, and then outputs a cleaned csv to data/enron.csv.
# The cleaned dataset is optimized for author identification. The fields are author, subject, recipients, and content.
# Author is the email address of the sender.
# Subject is the subject field of the email.
# Recipients is a space separated list of email address for all recipients of the email.
# Content is the content of the email. Note that most emails include a long chain of previous replies. These are removed and only the content of the current email is maintained. Keeping all those previous replies would make training harder as the content of every email would consist largely of emails from other authors.

import pandas as pd
import kagglehub
import os


def main():
    path = (
        kagglehub.dataset_download("wcukierski/enron-email-dataset")
        + os.sep
        + "emails.csv"
    )
    df = pd.read_csv(path)

    rows = []
    for id, series in df.iterrows():
        if id % 10_000 == 0:
            print(f"{id}/{len(df)}")
        assert (
            series["message"][:11] == "Message-ID:"
        ), 'Error reading dataset: message field did not start with "Message-ID:"'
        rows.append(parse_message(series["message"]))
    output_df = pd.DataFrame(
        rows, columns=["author", "recipients", "subject", "content"]
    )
    output_df = output_df[output_df["content"] != "N/A"]
    output_df = output_df[output_df["content"] != ""]
    output_df = output_df.reindex()
    os.makedirs("data", exist_ok=True)
    output_df.to_csv("./data/emails.csv", index=False)


def parse_message(message: str) -> tuple[str, str, str, str]:
    # Tuples of message fields is as follows. (Author, Recipients, Subject, Content)
    return (
        parse_author(message),
        parse_recipients(message),
        parse_subject(message),
        parse_content(message),
    )


def parse_author(message: str) -> str:
    from_line = message.splitlines()[2]
    assert from_line[0:6] == "From: ", "Could not parse author line"
    return from_line[6:]


def parse_subject(message: str) -> str:
    for line in message.splitlines():
        if line[:9] == "Subject: ":
            return line[9:]
    assert False, "Could not find subject line"


def parse_recipients(message: str) -> str:
    lines = message.splitlines()

    recipients = ""
    start_line = None
    end_line = None

    # Get start and end lines for two field
    for idx, line in enumerate(lines):
        if line[:4] == "To: ":
            start_line = idx
        if line[:9] == "Subject: ":
            end_line = idx
            break
    recipient_lines = lines[start_line:end_line]
    recipient_lines[0] = recipient_lines[0][4:]

    # Parse the email addresses from each line
    for line in recipient_lines:
        line = line.strip()
        recipients_on_line = line.split(",")
        for rec in recipients_on_line:
            stripped = rec.strip()
            if stripped == "":
                continue
            recipients += f"{stripped} "
    return recipients.strip()


def parse_content(message: str) -> str:
    lines = message.splitlines()

    # X-FileName appears to be the last line of meta data
    x_file_name_line_number = get_x_file_name_line_number(lines)
    # Make sure the next line is empty
    assert (
        lines[x_file_name_line_number + 1].strip() == ""
    ), "Did not find empty line after X-FileName line."

    if is_forwarded(lines[x_file_name_line_number + 2 :]):
        return "N/A"

    content = get_content_from_lines(lines[x_file_name_line_number + 2 :])
    if content.strip == "":
        return "N/A"
    return content


def get_x_file_name_line_number(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        if line[0:12] == "X-FileName: ":
            return index
    assert False, "Could not find X-FileName: line"


def is_forwarded(lines: list[str]) -> bool:
    while lines[0].strip() == "":
        lines = lines[1:]
    return "-forwarded" in lines[0].lower().replace(" ", "")


def get_content_from_lines(lines: list[str]) -> str:
    content = ""
    for line in lines:
        if "-original" in line.lower().replace(" ", ""):
            break
        if "-forward" in line.lower().replace(" ", ""):
            break
        content += line + "\n"
    return content


if __name__ == "__main__":
    main()
