def ctc_decode(self, inds: List[int]) -> str:
    # first we filter ctc tokens
    if isinstance(inds, Tensor):
        inds = inds.tolist()
    new_inds = []
    for ind in inds:
        if len(new_inds) and new_inds[-1] == ind:
            continue
        if len(new_inds) and new_inds[-1] == self.char2ind[self.EMPTY_TOK]:
            new_inds.pop()
        new_inds.append(self.ind2char[ind])

    if new_inds[-1] == self.char2ind[self.EMPTY_TOK]:
        new_inds.pop()

    return self.bpe.decode(new_inds)
