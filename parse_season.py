import numpy, os, sys, matplotlib, datetime
# matplotlib.use("GTK")
import pylab
from operator import itemgetter
from L2regression import LogisticRegression
from math import exp, log
from scipy.optimize import leastsq, fmin

if len(sys.argv) != 4:
    print >>sys.stderr, "usage: python %s cbbgaXX.txt compareXX.txt RANK" % sys.argv[0]
    sys.exit(1)

f = open(sys.argv[1], 'r')

# some flags...
accOnly = False
useLogReg = False
K = 2.5 # sigmoid points to probability scaling constant...
A = 0.0 # alpha L2 regularization penalty for LR
dateDamping = True
DDF = 0.5 # date damping factor: the earliest game margin is weighted by this amount, linearly increasing to 1.0 for the last game
useNeutrality = False # whether or not to include a home game indicator variable
LAMBDA = 1e-6 # L2 regularization penalty for linear least squares model (pinv)
useFactorize = True # do matrix factorization method and see what happens...
D = int(sys.argv[3]) # reduced rank for matrix factorization
earlyPred = True # include all teams, not just NCAA teams... because we have no tournament results

games = {}
gameCounts = {}

scoretotal = 0.0
count = 0.0

tournGames = {}

acc = set(["Boston College", "Clemson", "Duke", "Florida St.", "Georgia Tech", "Maryland",
           "Miami FL", "North Carolina", "North Carolina St.", "Virginia", "Virginia Tech", "Wake Forest"])

mindate = None
maxdate = -1000
lastTeams = None

for line in f:
    if len(line.strip()) <= 0: continue
    # team1 is the home team... unless it was neutral... we can check later...
    date, team2, score2, team1, score1, type = (line[:10].strip(),
                                                line[11:34].strip(),
                                                int(line[34:38]),
                                                line[38:61].strip(),
                                                int(line[61:65]),
                                                line[65:69].strip())
    
    mm, dd, yy = int(date[0:2]), int(date[3:5]), int(date[6:10])
    date = datetime.date(yy, mm, dd).toordinal()
    if mindate is None:
        mindate = date
    date -= mindate
    target = games
    if type.find("T") != -1:
        if accOnly:
            target = tournGames
            date = maxdate
    if type.find("P") != -1:
        if accOnly:
            continue
        else:            
            pass
        # ALERT: changed this
            # target = tournGames
            # print team1, team2
            # date = maxdate

    if date > maxdate:
        maxdate = date

    if accOnly and (not team1 in acc or not team2 in acc):
        continue

    neutral = False
    if type.find("N") != -1:
        neutral = True

    if not target.has_key(team1):
        target[team1] = {}

    target[team1][team2] = (score1, score2, neutral, date)

    if not gameCounts.has_key(team1):
        gameCounts[team1] = 0
    if not gameCounts.has_key(team2):
        gameCounts[team2] = 0
    gameCounts[team1] += 1
    gameCounts[team2] += 1

    lastTeams = team1, team2 # assuming input is in chronological order

    if score1 < 10 or score2 < 10 or score1 > 170 or score2 > 170: # sanity checks for parsing
        print line
        print score1, score2
    scoretotal += score1+score2
    count += 2
f.close()

print "average points scored: %.2f" % (scoretotal/count)

# print games["Duke"].has_key("Connecticut")
# print games["Connecticut"].has_key("Duke")
# print len(games["Duke"])
# print len(games)

# prune teams
kill = set()
for team, count in gameCounts.iteritems():
    if count < 0: #TODO: decrease this?
        kill.add(team)

if accOnly:
    kill = set()

for team in kill:
    games.pop(team, None)

for team1, v in games.iteritems():
    for team2 in v.iterkeys():
        if not games.has_key(team2):
            kill.add(team2)

    for team2 in kill:
        v.pop(team2, None)

print "killed %d teams" % len(kill)

# find teams in the NCAA tournament, not the NIT
tournTeams = set()
tournTeams.add(lastTeams[0])
tournTeams.add(lastTeams[1])

while True:
    change = False
    for team1, opp in tournGames.iteritems():
        for team2 in opp.iterkeys():
            if team1 in tournTeams or team2 in tournTeams:
                if not team1 in tournTeams:
                    change = True
                    # print "adding %s because they played %s" % (team1, team2)
                if not team2 in tournTeams:
                    change = True
                    # print "adding %s because they played %s" % (team2, team1)
                tournTeams.add(team1)
                tournTeams.add(team2)

    if not change: break

# NOTE: uncomment this if we have multiple tournaments in the dataset
#kill2 = set()
#for team1, opp in tournGames.iteritems():
#    if not team1 in tournTeams:
#        kill2.add(team1)
#for team in kill2:
#    tournGames.pop(team, None)

if earlyPred:
    tournTeams = set(games.keys())

# comment this out if we really have (historical) NCAA tournament results
# tournTeams = set(tournGames.keys())

# print sorted(list(tournTeams))
print "identified %d NCAA tournament teams" % len(tournTeams)
print tournTeams

# sys.exit(0)z
    
count = 0
indices = {}
f = open("team_indices.txt", 'w')
dotout = open("cyto_raw.out", 'w')
# print >>dotout, "graph G {"

for team in games.iterkeys():
    if team in kill: print "WARNING"
    indices[team] = count
    print >>f, count+1, team.replace(" ", "_")

    for team2 in games[team].iterkeys():
        # print >>dotout, "\t%s -- %s;" % (
        print >>dotout, "%s %s" % (
            team.replace(" ", "_").replace(".", "").replace("&", "").replace("'", ""), 
            team2.replace(" ", "_").replace(".", "").replace("&", "").replace("'", ""))
    count += 1
f.close()
print >>dotout, "}"
dotout.close()

train = []
targets = []
gameIndices = []

train.append([i+1 for i in xrange(len(games)+(2 if useNeutrality else 1))])
for team1 in games.iterkeys():
    for team2, result in games[team1].iteritems():
        vector = numpy.zeros(len(games)+(2 if useNeutrality else 1))
        vector[indices[team1]] = 1.0
        vector[indices[team2]] = -1.0
        target = result[0] - result[1]
        if dateDamping:
            if result[3] > maxdate: print >>sys.stderr, "WARNING"
            target *= (1.0-DDF)/maxdate * result[3] + DDF
        if useLogReg:
            target = 1.0/(1.0+exp(-target/K))
            # target = (numpy.sign(target)+2)/2
        else:
            # pass
            target = min(20, target)
            target = max(-20, target)

        targets.append(target)
        if useNeutrality:
            vector[-2] = not result[2]
        vector[-1] = target
        train.append(vector)

        if target != target:
            print >>sys.stderr, "WARNING"

        center = numpy.average((result[0], result[1]))
        # center = 50.0
        gameIndices.append((indices[team1], indices[team2], center+target))
        gameIndices.append((indices[team2], indices[team1], center-target))

print gameIndices

if useFactorize:
    offenses = numpy.zeros((len(games), D))
    offenses += 5.0
    defenses = numpy.zeros((len(games), D))
    defenses += 5.0

    # def predict(index1, index2):
    #    return numpy.dot(offenses[index1,:], defenses[index2,:]) # - numpy.dot(offenses[index2,:], defenses[index1,:])

    def predictNames(team1, team2):
        return predict(indices[team1], indices[team2]) - predict(indices[team2], indices[team1])

    def predictNamesFull(team1, team2):
        return ("%.2f" % predict(indices[team1], indices[team2]), "%.2f" % predict(indices[team2], indices[team1]))

    """
    for k in xrange(1):
        for i in xrange(D):
            for j in xrange(5): # iterations on each feature
                for index1, index2, target in gameIndices:
                    temp = offenses[index1]
                    err = (target - predict(index1, index2))
                    if err != err:
                        print err, i, j, index1, index2, predict(index1, index2), target, offenses[index1,:], defenses[index2,:]
                        sys.exit(0)
                    offenses[index1] += 0.001 * (err * defenses[index2] - 0.02 * offenses[index1])
                    defenses[index2] += 0.001 * (err * temp - 0.02 * defenses[index2])
    """
    def doOutput():
        powers1 = {}
        powers2 = {}
        powers3 = {}
        for team in games.iterkeys():
            if team in tournTeams:
                powers1[team] = tuple(offenses[indices[team],:])
                powers2[team] = tuple(defenses[indices[team],:])
                if D > 1:
                    powers3[team] = offenses[indices[team],1] / defenses[indices[team],1]
                elif D == 1:
                    powers3[team] = offenses[indices[team],0] - defenses[indices[team],0]
        #print "offenses (tourn. teams only):"
        #count = 1
        #for k,v in sorted(powers1.iteritems(), key=itemgetter(1), reverse=True)[:100]:
        #    print "%-3d %-20s %s" % (count, k,["%.2f" % x for x in v])
        #    count += 1
        #print "defenses (tourn. teams only):"
        #count = 1
        #for k,v in sorted(powers2.iteritems(), key=itemgetter(1), reverse=False)[:100]:
        #    print "%-3d %-20s %s" % (count, k,["%.2f" % x for x in v])
        #    count += 1
        if D == 1:
            print "O-D score (tourn. teams only):"
        else:
            print "O/D ratios (tourn. teams only):"
        count = 1
        ret = {}
        for k,v in sorted(powers3.iteritems(), key=itemgetter(1), reverse=True)[:500]:
            print "%-3d %-20s %.3f" % (count, k,v)
            ret[k] = count
            count += 1

        return ret

    # doOutput()

    """
    def targFunc(x):
        ret = numpy.zeros(len(gameIndices))
        count = 0
        for index1, index2, target in gameIndices:
            ret[count] = target - numpy.dot(x[index1], x[len(x)/2+index2])
            count += 1
        return ret
    
    print "optimizing with leastsq..."
    xhat = leastsq(targFunc, list(offenses.flatten())+list(defenses.flatten()), full_output=1, maxfev=2)
    print xhat
    sys.exit(0)
    """

    print "training"
    from rsvd.rsvd import RSVD, rating_t
    svdTrain = numpy.array(gameIndices, dtype=rating_t)
    print len(gameIndices)/2
    for i in xrange(len(gameIndices)):
        svdTrain[i][0] += 1
    model = RSVD.train(D, svdTrain, (len(indices), len(indices)),
                       regularization=0.0011, randomize=False, maxEpochs=100) #, globalMean=69.0)
    os.system("rm rsvd_output/*") # lazy and gross...
    model.save("rsvd_output")
    offenses = numpy.fromfile("rsvd_output/u.arr")
    defenses = numpy.fromfile("rsvd_output/v.arr")
    offenses = offenses.reshape(len(offenses)/D,D)
    defenses = defenses.reshape(len(defenses)/D,D)
    print numpy.shape(offenses)
    print "RSVD output:"
    myPowers = doOutput()
    
    """
    for team in acc:
        print team
    print "offense"
    for team in acc:
        print offenses[indices[team]][0], offenses[indices[team]][1]
    print "defense"
    for team in acc:
        print defenses[indices[team]][0], defenses[indices[team]][1]
    """

    def predict(index1, index2):
        return model(index1+1, index2)

train = numpy.array(train)

# numpy.savetxt("train.csv", train, fmt="%.0f", delimiter=",")

# pylab.hist(targets, numpy.arange(0,1.05,0.1))
print "score avg: %.3f and std. dev.: %.3f" % (numpy.average(targets), numpy.std(targets))
# pylab.show()

train = numpy.array(train)

if useLogReg: # logistic regression
    print "training"
    features = train[1:,:-1]
    regtargets = train[1:,-1]
    # print features
    # print regtargets
    lr = LogisticRegression(features, regtargets, alpha=A)
    lr.train()
    numpy.savetxt("betas.csv", lr.betas, fmt="%.6f", delimiter=",")
    bla = lr.betas
    print "done"
else:
    pass
    # bla = pylab.pinv(train[1:,:-1] + LAMBDA*numpy.eye(*numpy.shape(train[1:,:-1])))
    # bla = numpy.array(numpy.matrix(bla) * numpy.transpose(numpy.matrix(targets)))[:,0]

if useNeutrality:
    homeAdv = bla[-1]
    print "home team advantage: %.3f" % homeAdv
else:
    homeAdv = 0

powers = {}

for team in games.iterkeys():
    powers[team] = 0 # bla[indices[team]]

count = 1
for k,v in sorted(powers.iteritems(), key=itemgetter(1), reverse=True)[:0]:
    print "%-3d %-20s %.3f" % (count, k,v)
    count += 1

"""
correct, wrong = 0, 0
scorediffs = []
for team1, season in games.iteritems():
    for team2, result in season.iteritems():
        predicted = powers[team1] - powers[team2]
        scorediffs.append((result[0]-result[1]) - predicted)
        if numpy.sign(result[0] - result[1]) == numpy.sign(powers[team1] - powers[team2] + (homeAdv if not result[2] else 0)):
            correct += 1
        else:
            wrong += 1

if correct + wrong > 0 and correct > 0:
    print "\nregular season prediction: %d correct, %d wrong, (%.2f pct)" % (correct, wrong, 100.0*correct/(correct+wrong))
    # print >>sys.stderr, 1.0*correct/(correct+wrong)
    print "MSE point differential: %.3f" % numpy.std(scorediffs)

### copied code alert...

correct, wrong = 0, 0
scorediffs = []
for team1, season in tournGames.iteritems():
    for team2, result in season.iteritems():
        predicted = powers[team1] - powers[team2]
        scorediffs.append((result[0]-result[1]) - predicted)
        if numpy.sign(result[0] - result[1]) == numpy.sign(powers[team1] - powers[team2] + (homeAdv if not result[2] else 0)):
            correct += 1
        else:
            # print "missed %s vs. %s," % (team1, team2),
            wrong += 1
print ""
if correct + wrong > 0 and correct > 0:
    print "\npostseason prediction: %d correct, %d wrong, (%.2f pct)" % (correct, wrong, 100.0*correct/(correct+wrong))
    print "MSE point differential: %.3f" % numpy.std(scorediffs)
"""

### double copied code alert...
correct, wrong = 0, 0
scorediffs = []
for team1, season in games.iteritems():
    for team2, result in season.iteritems():
        predicted = predictNames(team1, team2)
        scorediffs.append((result[0]-result[1]) - predicted)
        if numpy.sign(result[0] - result[1]) == numpy.sign(predictNames(team1, team2)):
            correct += 1
        else:
            wrong += 1

if correct + wrong > 0:
    print "\nRRMF regular season prediction: %d correct, %d wrong, (%.2f pct)" % (correct, wrong, 100.0*correct/(correct+wrong))
    print >>sys.stderr, D, 1.0*correct/(correct+wrong),
    print "MSE point differential: %.3f" % numpy.std(scorediffs)
    print >>sys.stderr, numpy.std(scorediffs)

correct, wrong = 0, 0
scorediffs = []
for team1, season in tournGames.iteritems():
    for team2, result in season.iteritems():
        predicted = predictNames(team1, team2)
        scorediffs.append((result[0]-result[1]) - predicted)
        if numpy.sign(result[0] - result[1]) == numpy.sign(predictNames(team1, team2)):
            correct += 1
            # print "got %s vs. %s [real: %s, pred: %s]" % (team1, team2, (result[0],result[1]), predictNamesFull(team1, team2))
        else:
            # print "  missed %s vs. %s [real: %s, pred: %s]" % (team1, team2, (result[0],result[1]), predictNamesFull(team1, team2))
            wrong += 1
print ""
if correct + wrong > 0:
    print "\nRRMF postseason prediction: %d correct, %d wrong, (%.2f pct)" % (correct, wrong, 100.0*correct/(correct+wrong))
    print >>sys.stderr, D, 1.0*correct/(correct+wrong),
    print "MSE point differential: %.3f" % numpy.std(scorediffs)
    print >>sys.stderr, numpy.std(scorediffs)

### end copied code alert

winners = {}
for team1, opp in tournGames.iteritems():
    if not winners.has_key(team1):
        winners[team1] = 0
    for team2, result in opp.iteritems():
        if not winners.has_key(team2):
            winners[team2] = 0
        if result[0] > result[1]:
            winners[team1] += 1
        else:
            winners[team2] += 1

"""
finalfour = sorted(winners.iteritems(), key=itemgetter(1), reverse=True)[:4]
print finalfour

print "RRMF:", myPowers[finalfour[0][0]], myPowers[finalfour[1][0]], myPowers[finalfour[2][0]], myPowers[finalfour[3][0]]
"""

def reportOther(name, verbose=True, stopEarly=False):
    print ""
    from parse_rankings import get_ranks
    powers = get_ranks(sys.argv[2], name, negate=True)
    if len(powers) < 10: return False
    print "loaded %d teams for %s" % (len(powers), name)
    print -powers[finalfour[0][0]], -powers[finalfour[1][0]], -powers[finalfour[2][0]], -powers[finalfour[3][0]]
    if stopEarly:
        return True
    correct, wrong = 0, 0
    scorediffs = []
    for team1, season in tournGames.iteritems():
        for team2, result in season.iteritems():
            predicted = powers[team1] - powers[team2]
            scorediffs.append((result[0]-result[1]) - predicted)
            if numpy.sign(result[0] - result[1]) == numpy.sign(predicted):
                correct += 1
            else:
                if verbose:
                    print "  %s missed %s vs. %s" % (name, team1, team2)
                wrong += 1
    # print ""
    if correct + wrong > 0:
        print "%s postseason prediction: %d correct, %d wrong, (%.2f pct)" % (name, correct, wrong, 100.0*correct/(correct+wrong))
        print "MSE point differential: %.3f" % numpy.std(scorediffs)
    return True

"""
reportOther("RPI", verbose=False)
reportOther("SAG", verbose=False)
if not reportOther("MB", verbose=False):
    reportOther("MAS", verbose=False)
reportOther("USA", verbose=False, stopEarly=True)
reportOther("AP", verbose=False, stopEarly=True)

reportOther("LMC", verbose=False, stopEarly=False)
"""
