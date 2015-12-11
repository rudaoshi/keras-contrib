__author__ = 'Sun'


import numpy as np



def generate_text(corpus, rnn_model, starting_chars, random):


    line_break_id = corpus.get_id("\n")
    print "Line break id = ", line_break_id
    for char in starting_chars:

        char_id = corpus.get_id(char)
        char_ids = []
        char_ids.append(char_id)

        while char_id != line_break_id:

            next_char_prob = rnn_model.predict(np.array([[char_id]]))[0][0]
            next_char_prob = next_char_prob/sum(next_char_prob)
            if random:
                next_char_sample = np.random.multinomial(1, next_char_prob, size=1)
                next_char_id = np.where(next_char_sample[0] == 1)[0][0]
            else:
                next_char_id = np.argmax(next_char_prob)

            print "Next break id = ", next_char_id
            char_ids.append(next_char_id)

            char_id = next_char_id

        text = "".join(corpus.get_char(id) for id in char_ids)

        yield text

import cPickle
import click
@click.command()
@click.argument("char_cropus_file", type=click.File(mode='rb'))
@click.argument("model_file", type=click.File(mode='rb'))
@click.argument("starting_char_file", type=click.File(mode='r', encoding="gb18030"))
@click.argument("text_file", type=click.File(mode='w', encoding="gb18030"))
@click.option("--random/--no-random", default=True)
@click.option("--exp", type=click.INT, default=1)
def generate_text_interface(char_cropus_file, model_file,
                    starting_char_file, text_file, random, exp):

    corpus = cPickle.load(char_cropus_file)
    model = cPickle.load(model_file)
    starting_chars = starting_char_file.read().strip()

    for i in range(exp):
        for text in generate_text(corpus, model, starting_chars, random):
            text_file.write(text)

if __name__ == "__main__":

    generate_text_interface()





