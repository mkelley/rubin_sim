import matplotlib
matplotlib.use("Agg")
import numpy as np
import unittest
import rubin_sim.maf.metrics as metrics


class TestCadenceMetrics(unittest.TestCase):

    def testPhaseGapMetric(self):
        """
        Test the phase gap metric
        """
        data = np.zeros(10, dtype=list(zip(['observationStartMJD'], [float])))
        data['observationStartMJD'] += np.arange(10)*.25

        pgm = metrics.PhaseGapMetric(nPeriods=1, periodMin=0.5, periodMax=0.5)
        metricVal = pgm.run(data)

        meanGap = pgm.reduceMeanGap(metricVal)
        medianGap = pgm.reduceMedianGap(metricVal)
        worstPeriod = pgm.reduceWorstPeriod(metricVal)
        largestGap = pgm.reduceLargestGap(metricVal)

        self.assertEqual(meanGap, 0.5)
        self.assertEqual(medianGap, 0.5)
        self.assertEqual(worstPeriod, 0.5)
        self.assertEqual(largestGap, 0.5)

        pgm = metrics.PhaseGapMetric(nPeriods=2, periodMin=0.25, periodMax=0.5)
        metricVal = pgm.run(data)

        meanGap = pgm.reduceMeanGap(metricVal)
        medianGap = pgm.reduceMedianGap(metricVal)
        worstPeriod = pgm.reduceWorstPeriod(metricVal)
        largestGap = pgm.reduceLargestGap(metricVal)

        self.assertEqual(meanGap, 0.75)
        self.assertEqual(medianGap, 0.75)
        self.assertEqual(worstPeriod, 0.25)
        self.assertEqual(largestGap, 1.)

    def testTemplateExists(self):
        """
        Test the TemplateExistsMetric.
        """
        names = ['finSeeing', 'observationStartMJD']
        types = [float, float]
        data = np.zeros(10, dtype=list(zip(names, types)))
        data['finSeeing'] = [2., 2., 3., 1., 1., 1., 0.5, 1., 0.4, 1.]
        data['observationStartMJD'] = np.arange(10)
        slicePoint = {'sid': 0}
        # so here we have 4 images w/o good previous templates
        metric = metrics.TemplateExistsMetric(seeingCol='finSeeing')
        result = metric.run(data, slicePoint)
        self.assertEqual(result, 6./10.)

    def testUniformityMetric(self):
        names = ['observationStartMJD']
        types = [float]
        data = np.zeros(100, dtype=list(zip(names, types)))
        metric = metrics.UniformityMetric()
        result1 = metric.run(data)
        # If all the observations are on the 1st day, should be 1
        self.assertEqual(result1, 1)
        data['observationStartMJD'] = data['observationStartMJD']+365.25*10
        slicePoint = {'sid': 0}
        result2 = metric.run(data, slicePoint)
        # All on last day should also be 1
        self.assertEqual(result2, 1)
        # Make a perfectly uniform dist
        data['observationStartMJD'] = np.arange(0., 365.25*10, 365.25*10/100)
        result3 = metric.run(data, slicePoint)
        # Result should be zero for uniform
        np.testing.assert_almost_equal(result3, 0.)
        # A single obseravtion should give a result of 1
        data = np.zeros(1, dtype=list(zip(names, types)))
        result4 = metric.run(data, slicePoint)
        self.assertEqual(result4, 1)

    def testTGapMetric(self):
        names = ['observationStartMJD']
        types = [float]
        data = np.zeros(100, dtype=list(zip(names, types)))
        # All 1-day gaps
        data['observationStartMJD'] = np.arange(100)

        metric = metrics.TgapsMetric(bins=np.arange(1, 100, 1))
        result1 = metric.run(data)
        # By default, should all be in first bin
        self.assertEqual(result1[0], data.size-1)
        self.assertEqual(np.sum(result1), data.size-1)
        data['observationStartMJD'] = np.arange(0, 200, 2)
        result2 = metric.run(data)
        self.assertEqual(result2[1], data.size-1)
        self.assertEqual(np.sum(result2), data.size-1)

        data = np.zeros(4, dtype=list(zip(names, types)))
        data['observationStartMJD'] = [10, 20, 30, 40]
        metric = metrics.TgapsMetric(allGaps=True, bins=np.arange(1, 100, 10))
        result3 = metric.run(data)
        self.assertEqual(result3[1], 2)
        Ngaps = np.math.factorial(data.size-1)
        self.assertEqual(np.sum(result3), Ngaps)

    def testTGapsPercentMetric(self):
        names = ['observationStartMJD']
        types = [float]
        data = np.zeros(100, dtype=list(zip(names, types)))

        # All 1-day gaps
        data['observationStartMJD'] = np.arange(100)
        # All intervals are one day, so should be 100% within the gap specified
        metric = metrics.TgapsPercentMetric(minTime=0.5, maxTime=1.5, allGaps=False)
        result1 = metric.run(data)
        self.assertEqual(result1, 100)

        # All 2 day gaps
        data['observationStartMJD'] = np.arange(0, 200, 2)
        result2 = metric.run(data)
        self.assertEqual(result2, 0)

        # Run with allGaps = True
        data = np.zeros(4, dtype=list(zip(names, types)))
        data['observationStartMJD'] = [1, 2, 3, 4]
        metric = metrics.TgapsPercentMetric(minTime=0.5, maxTime=1.5, allGaps=True)
        result3 = metric.run(data)
        # This should be 50% -- 3 gaps of 1 day, 2 gaps of 2 days, 1 gap of 3 days
        self.assertEqual(result3, 50)

    def testNightGapMetric(self):
        names = ['night']
        types = [float]
        data = np.zeros(100, dtype=list(zip(names, types)))
        # All 1-day gaps
        data['night'] = np.arange(100)

        metric = metrics.NightgapsMetric(bins=np.arange(1, 100, 1))
        result1 = metric.run(data)
        # By default, should all be in first bin
        self.assertEqual(result1[0], data.size-1)
        self.assertEqual(np.sum(result1), data.size-1)
        data['night'] = np.arange(0, 200, 2)
        result2 = metric.run(data)
        self.assertEqual(result2[1], data.size-1)
        self.assertEqual(np.sum(result2), data.size-1)

        data = np.zeros(4, dtype=list(zip(names, types)))
        data['night'] = [10, 20, 30, 40]
        metric = metrics.NightgapsMetric(allGaps=True, bins=np.arange(1, 100, 10))
        result3 = metric.run(data)
        self.assertEqual(result3[1], 2)
        Ngaps = np.math.factorial(data.size-1)
        self.assertEqual(np.sum(result3), Ngaps)

        data = np.zeros(6, dtype=list(zip(names, types)))
        data['night'] = [1, 1, 2, 3, 3, 5]
        metric = metrics.NightgapsMetric(bins=np.arange(0, 5, 1))
        result4 = metric.run(data)
        self.assertEqual(result4[0], 0)
        self.assertEqual(result4[1], 2)
        self.assertEqual(result4[2], 1)

    def testNVisitsPerNightMetric(self):
        names = ['night']
        types = [float]
        data = np.zeros(100, dtype=list(zip(names, types)))
        # One visit per night.
        data['night'] = np.arange(100)

        bins = np.arange(0, 5, 1)
        metric = metrics.NVisitsPerNightMetric(bins=bins)
        result = metric.run(data)
        # All nights have one visit.
        expected_result = np.zeros(len(bins) - 1, dtype=int)
        expected_result[1] = len(data)
        np.testing.assert_array_equal(result, expected_result)

        data['night'] = np.floor(np.arange(0, 100) / 2)
        result = metric.run(data)
        expected_result = np.zeros(len(bins) - 1, dtype=int)
        expected_result[2] = len(data) / 2
        np.testing.assert_array_equal(result, expected_result)

    def testRapidRevisitUniformityMetric(self):
        data = np.zeros(100, dtype=list(zip(['observationStartMJD'], [float])))
        # Uniformly distribute time _differences_ between 0 and 100
        dtimes = np.arange(100)
        data['observationStartMJD'] = dtimes.cumsum()
        # Set up "rapid revisit" metric to look for visits between 5 and 25
        metric = metrics.RapidRevisitUniformityMetric(dTmin=5, dTmax=55, minNvisits=50)
        result = metric.run(data)
        # This should be uniform.
        self.assertLess(result, 0.1)
        self.assertGreaterEqual(result, 0)
        # Set up non-uniform distribution of time differences
        dtimes = np.zeros(100) + 5
        data['observationStartMJD'] = dtimes.cumsum()
        result = metric.run(data)
        self.assertGreaterEqual(result, 0.5)
        dtimes = np.zeros(100) + 15
        data['observationStartMJD'] = dtimes.cumsum()
        result = metric.run(data)
        self.assertGreaterEqual(result, 0.5)
        """
        # Let's see how much dmax/result can vary
        resmin = 1
        resmax = 0
        rng = np.random.RandomState(88123100)
        for i in range(10000):
            dtimes = rng.rand(100)
            data['observationStartMJD'] = dtimes.cumsum()
            metric = metrics.RapidRevisitUniformityMetric(dTmin=0.1, dTmax=0.8, minNvisits=50)
            result = metric.run(data)
            resmin = np.min([resmin, result])
            resmax = np.max([resmax, result])
        print("RapidRevisitUniformity .. range", resmin, resmax)
        """

    def testRapidRevisitMetric(self):
        data = np.zeros(100, dtype=list(zip(['observationStartMJD'], [float])))
        dtimes = np.arange(100)/24./60.
        data['observationStartMJD'] = dtimes.cumsum()
        # Set metric parameters to the actual N1/N2 values for these dtimes.
        metric = metrics.RapidRevisitMetric(dTmin=40./60./60./24., dTpairs=20./60./24.,
                                            dTmax=30./60./24., minN1=19, minN2=29)
        result = metric.run(data)
        self.assertEqual(result, 1)
        # Set metric parameters to > N1/N2 values, to see it return 0.
        metric = metrics.RapidRevisitMetric(dTmin=40.0/60.0/60.0/24., dTpairs=20./60./24.,
                                            dTmax=30./60./24., minN1=30, minN2=50)
        result = metric.run(data)
        self.assertEqual(result, 0)
        # Test with single value data.
        data = np.zeros(1, dtype=list(zip(['observationStartMJD'], [float])))
        result = metric.run(data)
        self.assertEqual(result, 0)

    def testNRevisitsMetric(self):
        data = np.zeros(100, dtype=list(zip(['observationStartMJD'], [float])))
        dtimes = np.arange(100)/24./60.
        data['observationStartMJD'] = dtimes.cumsum()
        metric = metrics.NRevisitsMetric(dT=50.)
        result = metric.run(data)
        self.assertEqual(result, 50)
        metric = metrics.NRevisitsMetric(dT=50., normed=True)
        result = metric.run(data)
        self.assertEqual(result, 0.5)

    def testTransientMetric(self):
        names = ['observationStartMJD', 'fiveSigmaDepth', 'filter']
        types = [float, float, '<U1']

        ndata = 100
        dataSlice = np.zeros(ndata, dtype=list(zip(names, types)))
        dataSlice['observationStartMJD'] = np.arange(ndata)
        dataSlice['fiveSigmaDepth'] = 25
        dataSlice['filter'] = 'g'

        metric = metrics.TransientMetric(surveyDuration=ndata/365.25)

        # Should detect everything
        self.assertEqual(metric.run(dataSlice), 1.)

        # Double to survey duration, should now only detect half
        metric = metrics.TransientMetric(surveyDuration=ndata/365.25*2)
        self.assertEqual(metric.run(dataSlice), 0.5)

        # Set half of the m5 of the observations very bright, so kill another half.
        dataSlice['fiveSigmaDepth'][0:50] = 20
        self.assertEqual(metric.run(dataSlice), 0.25)

        dataSlice['fiveSigmaDepth'] = 25
        # Demand lots of early observations
        metric = metrics.TransientMetric(peakTime=.5, nPrePeak=3, surveyDuration=ndata/365.25)
        self.assertEqual(metric.run(dataSlice), 0.)

        # Demand a reasonable number of early observations
        metric = metrics.TransientMetric(peakTime=2, nPrePeak=2, surveyDuration=ndata/365.25)
        self.assertEqual(metric.run(dataSlice), 1.)

        # Demand multiple filters
        metric = metrics.TransientMetric(nFilters=2, surveyDuration=ndata/365.25)
        self.assertEqual(metric.run(dataSlice), 0.)

        dataSlice['filter'] = ['r', 'g']*50
        self.assertEqual(metric.run(dataSlice), 1.)

        # Demad too many observation per light curve
        metric = metrics.TransientMetric(nPerLC=20, surveyDuration=ndata/365.25)
        self.assertEqual(metric.run(dataSlice), 0.)

        # Test both filter and number of LC samples
        metric = metrics.TransientMetric(nFilters=2, nPerLC=3, surveyDuration=ndata/365.25)
        self.assertEqual(metric.run(dataSlice), 1.)

    def testSeasonLengthMetric(self):
        times = np.arange(0, 3650, 10)
        data = np.zeros(len(times), dtype=list(zip(['observationStartMJD', 'visitExposureTime'], [float, float])))
        data['observationStartMJD'] = times
        data['visitExposureTime'] = 30.0
        metric = metrics.SeasonLengthMetric(reduceFunc=np.median)
        slicePoint = {'ra': 0}
        result = metric.run(data, slicePoint)
        self.assertEqual(result, 350)
        times = np.arange(0, 3650, 365)
        data = np.zeros(len(times)*2, dtype=list(zip(['observationStartMJD', 'visitExposureTime'], [float, float])))
        data['observationStartMJD'][0:len(times)] = times
        data['observationStartMJD'][len(times):] = times + 10
        data['observationStartMJD'] = np.sort(data['observationStartMJD'])
        data['visitExposureTime'] = 30.0
        metric = metrics.SeasonLengthMetric(reduceFunc=np.median)
        slicePoint = {'ra': 0}
        result = metric.run(data, slicePoint)
        self.assertEqual(result, 10)
        times = np.arange(0, 3650-365, 365)
        data = np.zeros(len(times)*2, dtype=list(zip(['observationStartMJD', 'visitExposureTime'], [float, float])))
        data['observationStartMJD'][0:len(times)] = times
        data['observationStartMJD'][len(times):] = times + 10
        data['observationStartMJD'] = np.sort(data['observationStartMJD'])
        data['visitExposureTime'] = 30.0
        metric = metrics.SeasonLengthMetric(reduceFunc=np.size)
        slicePoint = {'ra': 0}
        result = metric.run(data, slicePoint)
        self.assertEqual(result, 9)


if __name__ == "__main__":
    unittest.main()
