


class CellDict(object):

    def __init__(self, with_start = False, with_end = False, with_unk = False, with_pad=True):

        self.cell_id_map = dict()
        self.id_cell_map = dict()

        self.with_start = with_start
        self.with_end = with_end
        self.with_unk = with_unk


        if with_pad:
            # 0 is reserved for padding
            self.cell_id_map[""] = 0
            self.id_cell_map[0] = ""

        if with_start:
            id = len(self.cell_id_map)
            self.cell_id_map["<start>"] = id
            self.id_cell_map[id] = "<start>"

        if with_end:
            id = len(self.cell_id_map)
            self.cell_id_map["<eos>"] = id
            self.id_cell_map[id] = "<eos>"

        if with_unk:
            id = len(self.cell_id_map)
            self.cell_id_map["<unk>"] = id
            self.id_cell_map[id] = "<unk>"

    def iter_ids(self):

        for id in self.id_cell_map:
            yield id

    def iter_cell(self):

        for cell in self.cell_id_map:
            yield cell

    def id(self, cell):
        return self.cell_id_map[cell]

    def cell(self, id):
        return self.id_cell_map[id]

    def cell_num(self):
        return len(self.id_cell_map)

    def update(self, cells):

        cur_corpus = [0] * len(cells)

        for idx in range(len(cells)):
            cell = cells[idx]
            if cell not in self.cell_id_map:
                id = len(self.cell_id_map)
                self.cell_id_map[cell] = id
                self.id_cell_map[id] = cell
            else:
                id = self.cell_id_map[cell]

            cur_corpus[idx] = id

        if self.with_start:
            cur_corpus = [self.id("<start>")] + cur_corpus
        if self.with_end:
            cur_corpus = cur_corpus + [self.id("<eos>")]

        return cur_corpus


    def predict(self, cells):

        cur_corpus = [0] * len(cells)

        for idx in range(len(cells)):
            cell = cells[idx]
            if cell not in self.cell_id_map:
                if self.with_unk:
                    id = self.id("<unk>")
                else:
                    raise Exception("Unknown cell found. the repo should build with with_unk = True")
            else:
                id = self.cell_id_map[cell]

            cur_corpus[idx] = id

        if self.with_start:
            cur_corpus = [self.id("<start>")] + cur_corpus
        if self.with_end:
            cur_corpus = cur_corpus + [self.id("<eos>")]

        return cur_corpus