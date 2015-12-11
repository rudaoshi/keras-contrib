__author__ = 'Sun'


from sandbox.dynamic_title.creator.char_corpus import CharacterCorpus


import cPickle
import click
@click.command()
@click.argument("text_file", type=click.File(mode='r', encoding='gb18030'))
@click.argument("char_cropus_file", type=click.File(mode='wb'))
def make_char_corpus(text_file, char_cropus_file):

    corpus = CharacterCorpus()
    corpus.build(text_file)

    cPickle.dump(corpus, char_cropus_file, protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    make_char_corpus()