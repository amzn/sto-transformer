"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
"""

import string
from IMDB.Config import FLAG_OUTPUT


# rewrite print method to print only once when using multiple gpus
def one_print(info):
    if FLAG_OUTPUT:
        print(info)


def imdb_tokenize(input):
    """
        Naive tokenizer, that lower-cases the input
        and splits on punctuation and whitespace
    """
    input = input.lower()
    for p in string.punctuation:
        input = input.replace(p," ")
    return input.strip().split()
