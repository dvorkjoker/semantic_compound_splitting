#!/usr/bin/env python3
__author__ = 'lqrz'

#import cPickle as pickle
import pickle
import logging
import pdb
from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import WhitespaceTokenizer
import sys
import multiprocessing as mp
import codecs
from annoy import AnnoyIndex
from sklearn.metrics.pairwise import cosine_similarity
import gensim

def decompound(inputCompound, nAccuracy, similarityThreshold):
    global annoy_tree
    global vectors
    global model
    global globalNN

    # 1. See if we can deal with compound
    #
    if len(inputCompound) == 0:
        return []

    logger.debug('Looking up word %s in Word2Vec model' % inputCompound)
    if inputCompound not in model.vocab:  # We haven't vector representation for compound
        logger.debug('ERROR COULDNT FIND KEY %s IN WORD2VEC MODEL' % inputCompound)
        return []
    logger.debug('Found key in index dict for word %s' % inputCompound)
    inputCompoundRep = model[inputCompound]
    inputCompoundIndex = model.vocab[inputCompound].index

    # 2. See if we have prefixes of compound in vocabulary
    #
    logger.info('Getting all matching prefixes')
    prefixes = set()
    for prefix in vectors.keys():
        if len(inputCompound) > len(prefix) and inputCompound.startswith(prefix):
            prefixes.add(prefix)
    logger.debug('Possible prefixes: %r' % prefixes)
    if len(prefixes) == 0:  # cannot split
        return []
    
    # 3. Get all possible splits (so that we have representations for both prefix and tail)
    #
    logger.info('Getting possible splits')
    splits = set()

    FUGENLAUTE = ['', 'e', 'es']
    for prefix in prefixes:
        rest = inputCompound[len(prefix):]

        # get all possible tails
        possible_tails = []
        for fug in FUGENLAUTE:
            if rest.startswith(fug):
                possible_tails += [rest[len(fug):], rest[len(fug):].title()]

        for tail in possible_tails:
            logger.debug('Tail: %s' % tail)
            if tail not in model.vocab:  # we haven't representation for this tail
                logger.debug('Discarding split %s %s %s' % (inputCompound, prefix, tail))
                continue
            splits.add((prefix, tail))
            logger.debug('Considering split %s %s %s' % (inputCompound, prefix, tail))

    if len(splits) == 0:
        logger.error('Cannot decompound %s' % inputCompound)
        return []

    # 4. See if retrieved splits are good in terms of word embeddings
    #
    result = []
    logger.info('Applying direction vectors to possible splits')

    for prefix, tail in splits:
        logger.debug('Applying %d directions vectors to split %s %s' % (len(vectors[prefix]), prefix, tail))

        for origin, evidence in vectors[prefix]:
            logger.debug('Prefix %s by indexes %d and %d' % (prefix, origin[0], origin[1]))
            #if origin[0] not in pickledVectors or origin[1] not in pickledVectors:
            #    logger.debug('ERROR %d or %d NOT FOUND KEY IN VECTOR DICT' % (origin[0], origin[1]))
            #    continue
            dirVectorCompoundRepresentation = model[model.index2word[origin[0]]]
            dirVectorTailRepresentation = model[model.index2word[origin[1]]] 
            dirVectorDifference = dirVectorCompoundRepresentation - dirVectorTailRepresentation

            #logger.debug('Looking up tail index %d' % tailRepresentationIndex)
            #if tailRepresentationIndex not in pickledVectors:
            #    logger.debug('ERROR COULDNT FIND KEY %d IN VECTOR DICT' % tailRepresentationIndex)
            #    continue
            predictionRepresentation = model[tail] + dirVectorDifference

            logger.debug('Getting Annoy KNN')
            try:
                neighbours = annoy_tree.get_nns_by_vector(list(predictionRepresentation), globalNN)[:nAccuracy]
                logger.debug(neighbours)
            except:
                logger.error('Problem found when retrieving KNN for prediction representation')
                logger.error(list(predictionRepresentation))
                exit()

            # find rank
            rank = -1
            for i, nei in enumerate(neighbours):
                if nei == inputCompoundIndex:
                    rank = i
            if rank == -1:
                logger.debug('%d not found in neighbours. NO RANK. WONT SPLIT' % inputCompoundIndex)
                continue
            logger.debug('%d found in neighbours. Rank: %d' % (inputCompoundIndex, rank))

            # compare cosine against threshold
            similarity = cosine_similarity(predictionRepresentation, inputCompoundRep)[0][0]
            logger.debug('Computed cosine similarity: %f' % similarity)
            if similarity < similarityThreshold:
                logger.debug('%d has too small cosine similarity, discarding' % inputCompoundIndex)
                continue

            result.append((prefix, tail, origin[0], origin[1], rank, similarity))

    return result

vertices_count = 0
distances = []
edges = {}


def get_decompound_lattice(inputCompound, nAccuracy, similarityThreshold):
    global vertices_count
    global distances
    global edges

    # 1. Initialize
    #
    vertices_count = 2
    distances = [0, 1000.0] # distance to each vertix (for top sort)
    edges = {0: [(1, inputCompound, 0, 1.0)]}  # from: (to, label, rank, cosine)

    # 2. Make graph
    #
    def add_edges(from_, to, label):
        global vertices_count
        global distances
        global edges

        candidates = decompound(label, nAccuracy, similarityThreshold)
        for index, candidate in enumerate(candidates):
            prefix, tail, origin0, origin1, rank, similarity = candidate
           
            if from_ not in edges:
                edges[from_] = []
 
            edges[from_] += [(vertices_count, prefix, rank, similarity)]
            edges[vertices_count] = [(to, tail, 0, 1.0)]
          
            try: 
                distances.append(distances[from_] + (distances[to] - distances[from_]) * (1 + index) / (1 + len(candidates)))
            except:
                print(distances, from_, to, index, candidates)
                raise
            vertices_count += 1
 
            add_edges(from_, vertices_count - 1, prefix)
            add_edges(vertices_count - 1, to, tail)

    add_edges(0, 1, inputCompound)

    # 3. Top sort & output
    #
    vertices = zip(range(vertices_count), distances)
    vertices = sorted(vertices, key=lambda x: x[1])
    new_indexes = {}
    for index, vertix in enumerate(vertices):
        new_indexes[vertix[0]] = index
   
    lattice = '('  # '(\n'
    for index, vertix in enumerate(vertices):
        if vertix[0] not in edges:
            continue
        lattice += '('  # '  (\n'
        for edge in edges[vertix[0]]:
            lattice += '(\'%s\',%d,%f,%d),' % (edge[1], edge[2], edge[3], new_indexes[edge[0]] - index)
        lattice += '),'
    lattice += ')'
    return lattice


if __name__ == '__main__':

    # logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('')
    hdlr = logging.FileHandler('decompound_annoy.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.DEBUG)


    globalNN = 500
    annoyTreeFile = '../../model/tree.ann'
    w2vPath = '../../model/w2v_500_de.bin'
    resultsPath = '../../model/prototypes/dir_vecs_10_100.p'
    nAccuracy = 250
    similarityThreshold = .0
    vectors = pickle.load(open(resultsPath, 'rb'))
    annoy_tree = AnnoyIndex(500)
    annoy_tree.load(annoyTreeFile)
    model = gensim.models.Word2Vec.load_word2vec_format(w2vPath, binary=True) 
    print('Loaded!', file=sys.stderr)
    for line in sys.stdin:
        print(get_decompound_lattice(line.rstrip('\n'), nAccuracy, similarityThreshold))

    exit(0)

'''
    # resultsPath = 'results/dir_vecs_4_100.p'
    # annoyTreeFile = 'tree.ann'
    # pickledIndexes = pickle.load(open('decompoundIndexes.p','rb'))
    # pickledVectors = pickle.load(open('decompoundVectors.p','rb'))
    # corpusPath = './prueba.txt'
    # outPath = 'splits.txt'

    # multiprocessed = True
    # nWorkers = 4
    #TODO: define threshold
    # nAccuracy= 100

    globalNN = 500

    if len(sys.argv) == 11:
        resultsPath = sys.argv[1]
        # w2vPath = sys.argv[2]
        annoyTreeFile = sys.argv[2]
        corpusPath = sys.argv[3]
        pickledIndexesPath = sys.argv[4]
        pickledVectorsPath = sys.argv[5]
        multiprocessed = sys.argv[6] == 'True'
        nWorkers = sys.argv[7]
        outPath = sys.argv[8]
        nAccuracy = int(sys.argv[9])
        similarityThreshold = float(sys.argv[10])

    elif len(sys.argv)>1:
        print 'Error in params'
        exit()


    idx = corpusPath.rfind('/') + 1
    folder = corpusPath[0:idx]
    filename = corpusPath[idx:]

    logger.debug('Corpus folder: '+folder)
    logger.debug('Corpus filename: '+filename)

    corpus = PlaintextCorpusReader(folder, filename, word_tokenizer=WhitespaceTokenizer(), encoding='utf-8')
    inputCompounds = corpus.words()

    logger.debug('Words in corpus')
    logger.debug(inputCompounds)

    debug = False

    logger.info('Getting pickled direction vectors file')
    vectors = pickle.load(open(resultsPath, 'rb'))

    logger.info('Getting pickled indexes')
    pickledIndexes = pickle.load(open(pickledIndexesPath,'rb'))
    pickledVectors = pickle.load(open(pickledVectorsPath,'rb'))

    logger.info('Getting annoy tree')
    # model = gensim.models.Word2Vec.load_word2vec_format(w2vPath, binary=True)
    annoy_tree = AnnoyIndex(500)
    annoy_tree.load(annoyTreeFile)

    if multiprocessed:
        logger.info('Instantiating pool with '+str(nWorkers))
        pool = mp.Pool(processes=int(nWorkers))
        results = pool.map(decompound, zip(inputCompounds, [nAccuracy]*len(inputCompounds), \
                                           [similarityThreshold]*len(inputCompounds)))
    else:
        results = []
        for inputCompound in inputCompounds:
            # try:
            #     inputCompoundIndex = pickledIndexes[inputCompound]
            #     compoundRepresentation = pickledVectors[inputCompoundIndex]
            #     results.append(decompound((inputCompound, nAccuracy)))
            results.append(decompound((inputCompound, nAccuracy, similarityThreshold)))
            # except KeyError:
            #     logger.error('No word2vec representation for input compound'+inputCompound)
            #     # exit()
            #     results.append(inputCompound)


    print results

    fout = codecs.open(outPath, 'w', encoding='utf-8')

    for i, split in enumerate(results):
        fout.write(inputCompounds[i] + '\t' + ' '.join(split) + '\n')
        # for comp, decomp1, decomp2 in split:
            # fout.write(comp + '\t' + decomp1 + '\t' + decomp2 + '\n')

    fout.close()

    logger.info('End')
'''
