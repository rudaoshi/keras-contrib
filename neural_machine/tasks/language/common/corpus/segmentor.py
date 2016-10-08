
class CharacterSegmenter(object):

    def segment(self, line):

        return line


class SpaceSegmenter(object):

    def segment(self, line):

        return filter(None, line.split(" "))