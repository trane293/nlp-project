def longestCommonSubsequence(string, sub):
    if (len(string) < len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if (string[i - 1] == sub[j - 1]):
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]

def score_ROGUE(hyp, ref, beta=1.2):

    prec = []
    rec = []

    # split into tokens
    token_c = hyp.split(" ")

    # split into tokens
    token_r = ref.split(" ")
    # compute the longest common subsequence
    lcs = longestCommonSubsequence(token_r, token_c)
    prec.append(lcs/float(len(token_c)))
    rec.append(lcs/float(len(token_r)))

    prec_max = max(prec)
    rec_max = max(rec)

    if(prec_max!=0 and rec_max !=0):
        score = ((1 + beta**2)*prec_max*rec_max)/float(rec_max + beta**2*prec_max)
    else:
        score = 0.0
    return score