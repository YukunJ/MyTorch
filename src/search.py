import numpy as np


def GreedySearch(SymbolSets, y_probs):
    """Greedy Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    Returns
    ------
    forward_path: str
                the corresponding compressed symbol sequence i.e. without blanks
                or repeated symbols.

    forward_prob: scalar (float)
                the forward probability of the greedy path

    """
    # Follow the pseudocode from lecture to complete greedy search :-)
    S, T, B = y_probs.shape
    forward_lst = []
    current_prob = 1.0
    for t in range(T):
        max_id = 0
        max_prob = y_probs[0, t, 0]
        for s in range(S):
            if y_probs[s, t, 0] > max_prob:
                max_prob = y_probs[s, t, 0]
                max_id = s
        current_prob *= max_prob
        if max_id != 0:
            forward_lst.append(SymbolSets[max_id-1])
        else:
            forward_lst.append('-')
    filtered_lst = []
    for i in range(len(forward_lst)):
        if (i < len(forward_lst)-1 and forward_lst[i] == forward_lst[i+1]) or forward_lst[i] == '-':
            continue
        filtered_lst.append(forward_lst[i])
    forward_path = "".join(list(map(str, filtered_lst)))
    forward_prob = current_prob

    return forward_path, forward_prob


##############################################################################


def BeamSearch(SymbolSets, y_probs, BeamWidth):
    """Beam Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    BeamWidth: int
                Width of the beam.

    Return
    ------
    bestPath: str
            the symbol sequence with the best path score (forward probability)

    mergedPathScores: dictionary
                        all the final merged paths with their scores.

    """
    _, T, B = y_probs.shape
    
    def InitializePaths(SymbolSets, y):
        InitialBlankPathScore, InitialPathScore = dict(), dict()
        InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol = set(), set()
        # First push the blank into a path-ending-with-blank stack. No symbol has been invoked yet
        path = '-'
        InitialBlankPathScore[path] = y[0]
        InitialPathsWithFinalBlank.add(path)
        # Push rest of the symbols into a path-ending-with-symbol stack
        for idx, c in enumerate(SymbolSets):
            path = c
            InitialPathScore[path] = y[idx+1]
            InitialPathsWithFinalSymbol.add(path)
        return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore
    
    def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
        PrunedBlankPathScore, PrunedPathScore = dict(), dict()
        # First gather all the relevant scores
        scorelist = list()
        for p in PathsWithTerminalBlank:
            scorelist.append(BlankPathScore[p])
        for p in PathsWithTerminalSymbol:
            scorelist.append(PathScore[p])
        scorelist.sort(reverse=True)
        
        # Sort and find cutoff score that retains exactly BeamWidth paths
        cutoff = scorelist[-1]
        if BeamWidth < len(scorelist):
            cutoff = scorelist[BeamWidth-1]
        
        PrunedPathsWithTerminalBlank = set()
        for p in PathsWithTerminalBlank:
            if BlankPathScore[p] >= cutoff:
                PrunedPathsWithTerminalBlank.add(p)
                PrunedBlankPathScore[p] = BlankPathScore[p]
        
        PrunedPathsWithTerminalSymbol = set()
        for p in PathsWithTerminalSymbol:
            if PathScore[p] >= cutoff:
                PrunedPathsWithTerminalSymbol.add(p)
                PrunedPathScore[p] = PathScore[p]
        
        return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore
        
    def ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y):
        UpdatedPathsWithTerminalBlank = set()
        UpdatedBlankPathScore = dict()
        
        # First work on paths with terminal blanks
        #(This represents transitions along horizontal trellis edges for blanks)
        for p in PathsWithTerminalBlank:
            UpdatedPathsWithTerminalBlank.add(p)
            UpdatedBlankPathScore[p] = BlankPathScore[p] * y[0]
            
        # Then extend paths with terminal symbols by blanks
        for p in PathsWithTerminalSymbol:
            # If there is already an equivalent string in UpdatesPathsWithTerminalBlank
            # simply add the score. If not create a new entry
            if p in UpdatedPathsWithTerminalBlank:
                UpdatedBlankPathScore[p] += PathScore[p] * y[0]
            else:
                UpdatedPathsWithTerminalBlank.add(p)
                UpdatedBlankPathScore[p] = PathScore[p] * y[0]
        
        return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore
        
    def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSets, y):
        UpdatedPathsWithTerminalSymbol = set()
        UpdatedPathScore = dict()
        
        # First extend the paths terminating in blanks. This will always create a new sequence
        for p in PathsWithTerminalBlank:
            for idx, c in enumerate(SymbolSets):
                newpath = p + c # SymbolSet does not include blanks
                UpdatedPathsWithTerminalSymbol.add(newpath)
                UpdatedPathScore[newpath] = BlankPathScore[p] * y[idx+1]
        
        # Next work on paths with terminal symbols
        for p in PathsWithTerminalSymbol:
            # Extend the path with every symbol other than blank
            for idx, c in enumerate(SymbolSets):
                if c == p[-1]: # Horizontal transitions don’t extend the sequence
                    newpath = p
                else:
                    newpath = p + c
                if newpath in UpdatedPathsWithTerminalSymbol:
                    UpdatedPathScore[newpath] += PathScore[p] * y[idx+1]
                else:
                    UpdatedPathsWithTerminalSymbol.add(newpath)
                    UpdatedPathScore[newpath] = PathScore[p] * y[idx+1]
        return UpdatedPathsWithTerminalSymbol, UpdatedPathScore
        
    def MergeIdenticalPaths(PathsWithTerminalBlank, BlankPathScore, PathsWithTerminalSymbol, PathScore):
        # All paths with terminal symbols will remain
        MergedPaths = PathsWithTerminalSymbol
        FinalPathScore = PathScore
        
        # Paths with terminal blanks will contribute scores to existing identical paths from
        # PathsWithTerminalSymbol if present, or be included in the final set, otherwise
        for p in PathsWithTerminalBlank:
            if p in MergedPaths:
                FinalPathScore[p] += BlankPathScore[p]
            else:
                MergedPaths.add(p)
                FinalPathScore[p] = BlankPathScore[p]
        
        return MergedPaths, FinalPathScore
                
        
    # First time instant: Initialize paths with each of the symbols,
    # including blank, using score at time t=1
    NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = InitializePaths(SymbolSets, y_probs[:, 0, 0])

    # Subsequent time steps
    for t in range(1, T):
        # Prune the collection down to the BeamWidth
        PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = \
                Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore, BeamWidth)

        # First extend paths by a blank
        NewPathsWithTerminalBlank, NewBlankPathScore = ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y_probs[:, t, 0])

        # Next extend paths by a symbol
        NewPathsWithTerminalSymbol, NewPathScore = ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSets, y_probs[:, t, 0])
    
    # Merge identical paths differing only by the final blank
    mergedPaths, FinalPathScore = MergeIdenticalPaths(NewPathsWithTerminalBlank, NewBlankPathScore, NewPathsWithTerminalSymbol, NewPathScore)
    bestPath = "not existing"
    bestScore = float('-inf')
    for path, score in FinalPathScore.items():
        if score > bestScore:
            bestScore = score
            bestPath = path
    return bestPath, FinalPathScore
    
