"""
Test
----
This is a test
"""

#log# Automatic Logger file. *** THIS MUST BE THE FIRST LINE ***
#log# DO NOT CHANGE THIS LINE OR THE TWO BELOW
#log# opts = Struct({'__allownew': True, 'logfile': 'ipython_log.py'})
#log# args = []
#log# It is safe to make manual edits below here.
#log#-----------------------------------------------------------------------
import numpy as np
from rsvd import RSVD
print 'load ratings'
ratings=np.load('data/ratings_float.arr')

probeRatings=np.load('data/probe_ratings_float.arr')


model=RSVD.train(20,ratings,(17770,480189),probeRatings,100,randomize=False)
print "model trained..."

model.save("models/t_20_001_011_100")


