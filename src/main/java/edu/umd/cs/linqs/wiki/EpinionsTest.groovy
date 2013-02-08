package edu.umd.cs.linqs.wiki

import org.slf4j.Logger
import org.slf4j.LoggerFactory

import com.google.common.collect.Iterables

import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.VotedPerceptron;
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.LossBalancingType
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.NormScalingType;
import edu.umd.cs.psl.application.learning.weight.random.FirstOrderMetropolisRandOM
import edu.umd.cs.psl.application.learning.weight.random.HardEMRandOM
import edu.umd.cs.psl.application.learning.weight.random.IncompatibilityMetropolisRandOM;
import edu.umd.cs.psl.application.learning.weight.random.MetropolisRandOM;
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.core.*
import edu.umd.cs.psl.core.inference.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.ResultList
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.*
import edu.umd.cs.psl.evaluation.statistics.RankingScore
import edu.umd.cs.psl.evaluation.statistics.SimpleRankingComparator
import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.Model
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.GroundTerm
import edu.umd.cs.psl.model.argument.UniqueID
import edu.umd.cs.psl.model.atom.GroundAtom
import edu.umd.cs.psl.model.atom.RandomVariableAtom
import edu.umd.cs.psl.model.kernel.CompatibilityKernel
import edu.umd.cs.psl.model.parameters.Weight
import edu.umd.cs.psl.ui.loading.*
import edu.umd.cs.psl.util.database.Queries


/*
 * SET LEARNING ALGORITHMS
 * 
 * Options:
 * "MLE" (MaxLikelihoodMPE)
 * "MPLE" (MaxPseudoLikelihood)
 * "MM" (MaxMargin)
 * "FirstOrderRandOM" (FirstOrderMetropolisRandOM)
 * "IncompatibilityRandOM" (IncompatibilityMetropolisRandOM)
 */
methods = ["MLE"];

/* MLE/MPLE options */
vpStepCounts = [10]
vpStepSizes = [10]

/* MM options */
slackPenalties = [1, 10, 100]
lossBalancings = [LossBalancingType.NONE, LossBalancingType.CLASS_WEIGHTS, LossBalancingType.INVERSE_CLASS_WEIGHTS]
normScalings = [NormScalingType.NONE, NormScalingType.NUM_GROUNDINGS, NormScalingType.INVERSE_NUM_GROUNDINGS]

/* Metropolis RandOM options */
// TODO

Logger log = LoggerFactory.getLogger(this.class)

ConfigManager cm = ConfigManager.getManager()
ConfigBundle baseConfig = cm.getBundle("epinions")

boolean sq = false

/*
 * DEFINES EXPERIMENT CONFIGURATIONS
 */
Map<String, ConfigBundle> methodConfigs = new HashMap<String, ConfigBundle>();
for (String method : methods) {
	if (method.equals("MLE") || method.equals("MPLE")) {
		for (int vpStepCount : vpStepCounts) {
			for (double vpStepSize : vpStepSizes) {
				ConfigBundle newBundle = cm.getBundle("epinions");
				newBundle.addProperty("method", method);
				newBundle.addProperty(VotedPerceptron.NUM_STEPS_KEY, vpStepCount);
				newBundle.addProperty(VotedPerceptron.STEP_SIZE_KEY, vpStepSize);
				methodConfigs.put(((sq) ? "quad" : "linear") + "-" + method.toLowerCase() + "-" + vpStepCount + "-" + vpStepSize, newBundle);
			}
		}
	}
	else if (method.equals("MM")) {
		for (double slackPenalty : slackPenalties) {
			for (LossBalancingType lossBalancing : lossBalancings) {
				for (NormScalingType normScaling : normScalings) {
					ConfigBundle newBundle = cm.getBundle("epinions");
					newBundle.addProperty("method", method);
					newBundle.addProperty(MaxMargin.SLACK_PENALTY_KEY, slackPenalty);
					newBundle.addProperty(MaxMargin.BALANCE_LOSS_KEY, lossBalancing);
					newBundle.addProperty(MaxMargin.SCALE_NORM_KEY, normScaling);
					methodConfigs.put(((sq) ? "quad" : "linear") + "-mm-" + slackPenalty + "-" + lossBalancing.name().toLowerCase()
							+ "-" + normScaling.name().toLowerCase(), newBundle);
				}
			}
		}
	}
	else if (method.equals("FirstOrderRandOM") || method.equals("IncompatibilityRandOM")) {
		ConfigBundle newBundle = cm.getBundle("epinions");
		newBundle.addProperty("method", method);
		numSamples = 75
		burnIn = 20
		maxIter = 30;
		newBundle.addProperty(MetropolisRandOM.NUM_SAMPLES_KEY, numSamples);
		newBundle.addProperty(MetropolisRandOM.BURN_IN_KEY, burnIn);
		newBundle.addProperty(MetropolisRandOM.MAX_ITER_KEY, maxIter);
		methodConfigs.put(((sq) ? "quad" : "linear") + "-" + method.toLowerCase() + "-" + numSamples + "-" + burnIn
				+ "-" + maxIter, newBundle);
	}
	else
		throw new IllegalArgumentException("Unrecognized method: " + method);
}

/*
 * PRINTS EXPERIMENT CONFIGURATIONS
 */
for (Map.Entry<String, ConfigBundle> config : methodConfigs.entrySet())
	System.out.println("Config for " + config.getKey() + "\n" + config.getValue());

/*
 * INITIALIZES DATASTORE AND MODEL
 */
//def defaultPath = System.getProperty("java.io.tmpdir") + "/"
def defaultPath = "/scratch0/bach-icml13/"
String dbpath = baseConfig.getString("dbpath", defaultPath + "pslEpinions")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), baseConfig)

PSLModel m = new PSLModel(this, data)

/*
 * DEFINE MODEL
 */
m.add predicate: "knows", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "trusts", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "prior", types: [ArgumentType.UniqueID]

m.add rule: (knows(A,B) & knows(B,C) & knows(A,C) & trusts(A,B) & trusts(B,C) & (A - B) & (B - C) & (A - C)) >> trusts(A,C), weight: 1.0, squared: sq   //FFpp
m.add rule: (knows(A,B) & knows(B,C) & knows(A,C) & trusts(A,B) & ~trusts(B,C) & (A - B) & (B - C) & (A - C)) >> ~trusts(A,C), weight: 1.0, squared: sq //FFpm
m.add rule: (knows(A,B) & knows(B,C) & knows(A,C) & ~trusts(A,B) & trusts(B,C) & (A - B) & (B - C) & (A - C)) >> ~trusts(A,C), weight: 1.0, squared: sq //FFmp
m.add rule: (knows(A,B) & knows(B,C) & knows(A,C) & ~trusts(A,B) & ~trusts(B,C) & (A - B) & (B - C) & (A - C)) >> trusts(A,C), weight: 1.0, squared: sq //FFmm

m.add rule: (knows(A,B) & knows(C,B) & knows(A,C) & trusts(A,B) & trusts(C,B) & (A - B) & (B - C) & (A - C)) >> trusts(A,C), weight:1.0, squared: sq  //FBpp
m.add rule: (knows(A,B) & knows(C,B) & knows(A,C) & trusts(A,B) & ~trusts(C,B) & (A - B) & (B - C) & (A - C)) >> ~trusts(A,C), weight:1.0, squared: sq //FBpm
m.add rule: (knows(A,B) & knows(C,B) & knows(A,C) & ~trusts(A,B) & trusts(C,B) & (A - B) & (B - C) & (A - C)) >> ~trusts(A,C), weight:1.0, squared: sq //FBmp
m.add rule: (knows(A,B) & knows(C,B) & knows(A,C) & ~trusts(A,B) & ~trusts(C,B) & (A - B) & (B - C) & (A - C)) >> trusts(A,C), weight:1.0, squared: sq //FBmm

m.add rule: (knows(B,A) & knows(B,C) & knows(A,C) & trusts(B,A) & trusts(B,C) & (A - B) & (B - C) & (A - C)) >> trusts(A,C), weight:1.0, squared: sq   //BFpp
m.add rule: (knows(B,A) & knows(B,C) & knows(A,C) & trusts(B,A) & ~trusts(B,C) & (A - B) & (B - C) & (A - C)) >> ~trusts(A,C), weight:1.0, squared: sq //BFpm
m.add rule: (knows(B,A) & knows(B,C) & knows(A,C) & ~trusts(B,A) & trusts(B,C) & (A - B) & (B - C) & (A - C)) >> ~trusts(A,C), weight:1.0, squared: sq //BFmp
m.add rule: (knows(B,A) & knows(B,C) & knows(A,C) & ~trusts(B,A) & ~trusts(B,C) & (A - B) & (B - C) & (A - C)) >> trusts(A,C), weight:1.0, squared: sq //BFmm

m.add rule: (knows(B,A) & knows(C,B) & knows(A,C) & trusts(B,A) & trusts(C,B) & (A - B) & (B - C) & (A - C)) >> trusts(A,C), weight:1.0, squared: sq   //BBpp
m.add rule: (knows(B,A) & knows(C,B) & knows(A,C) & trusts(B,A) & ~trusts(C,B) & (A - B) & (B - C) & (A - C)) >> ~trusts(A,C), weight:1.0, squared: sq //BBpm
m.add rule: (knows(B,A) & knows(C,B) & knows(A,C) & ~trusts(B,A) & trusts(C,B) & (A - B) & (B - C) & (A - C)) >> ~trusts(A,C), weight:1.0, squared: sq //BBmp
m.add rule: (knows(B,A) & knows(C,B) & knows(A,C) & ~trusts(B,A) & ~trusts(C,B) & (A - B) & (B - C) & (A - C)) >> trusts(A,C), weight:1.0, squared: sq //BBmm

m.add rule: (knows(A,B) & knows(B,A) & trusts(A,B)) >> trusts(B,A), weight: 1.0, squared: sq
m.add rule: (knows(A,B) & knows(B,A) & ~trusts(A,B)) >> ~trusts(B,A), weight: 1.0, squared: sq

// two-sided prior
UniqueID constant = data.getUniqueID(0)
m.add rule: (knows(A,B) & prior(constant)) >> trusts(A,B), weight: 1.0, squared: sq
m.add rule: (knows(A,B) & trusts(A,B)) >> prior(constant), weight: 1.0, squared: sq

// save all default weights
Map<CompatibilityKernel,Weight> weights = new HashMap<CompatibilityKernel, Weight>()
for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
	weights.put(k, k.getWeight());

Partition fullKnows =  new Partition(0)
Partition fullTrusts = new Partition(1)

/*
 * LOAD DATA
 */
def dataPath = "./epinions/"
def inserter
inserter = data.getInserter(knows, fullKnows)
InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "knows.txt")
inserter = data.getInserter(trusts, fullTrusts)
InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "trusts.txt")

// number of folds
folds = 8

List<Partition> trustsPartitions = new ArrayList<Partition>(folds)
List<Partition> knowsPartitions = new ArrayList<Partition>(folds)
List<Partition> trainWritePartitions = new ArrayList<Partition>(folds)
List<Partition> testWritePartitions = new ArrayList<Partition>(folds)
List<Partition> trainPriorPartitions = new ArrayList<Partition>(folds)
List<Partition> testPriorPartitions = new ArrayList<Partition>(folds)

Random rand = new Random(0)

for (int i = 0; i < folds; i++) {
	knowsPartitions.add(i, new Partition(i + 2))
	trustsPartitions.add(i, new Partition(i + folds + 2))
	trainWritePartitions.add(i, new Partition(i + 2*folds + 2))
	testWritePartitions.add(i, new Partition(i + 3*folds + 2))
	trainPriorPartitions.add(i, new Partition(i + 4*folds + 2))
	testPriorPartitions.add(i, new Partition(i + 5*folds + 2))
}

List<Set<GroundingWrapper>> groundings = FoldUtils.splitGroundings(data, [trusts, knows], [fullTrusts, fullKnows], folds)
for (int i = 0; i < folds; i++) {
	FoldUtils.copy(data, fullKnows, knowsPartitions.get(i), knows, groundings.get(i))
	FoldUtils.copy(data, fullTrusts, trustsPartitions.get(i), trusts, groundings.get(i))
}



Map<String, List<Double []>> results = new HashMap<String, List<Double []>>()
for (String method : methodConfigs.keySet())
	results.put(method, new ArrayList<Double []>())

for (int fold = 0; fold < folds; fold++) {

	/*
	 * Training set:
	 * 	all 'knows' data except partition 'fold'
	 * 	all 'trusts' data except partition 'fold' and 'fold-1' (mod)
	 * 	target labels: trusts data in partition 'fold-1' (mod)
	 * 
	 * Test set:
	 * 	all 'knows' data in the whole dataset
	 * 	all 'trusts' data except partition 'fold'
	 *	target: predict 'trusts' of partition 'fold'
	 * 
	 */
	ArrayList<Partition> trainReadPartitions = new ArrayList<Partition>();
	ArrayList<Partition> testReadPartitions = new ArrayList<Partition>();
	int trainingTarget = (fold - 1) % folds
	if (trainingTarget < 0) trainingTarget += folds
	Partition trainLabelPartition = trustsPartitions.get(trainingTarget)
	for (int i = 0; i < folds; i++) {
		if (i != fold) {
			testReadPartitions.add(trustsPartitions.get(i)) // TEST: only hold out fold'th trust partition
			trainReadPartitions.add(knowsPartitions.get(i)) // TRAIN: only observe knows outside of fold
		}
		testReadPartitions.add(knowsPartitions.get(i)) // TEST: include all knows information

		if (i != trainingTarget && i != fold)
			trainReadPartitions.add(trustsPartitions.get(i))
	}

	trainReadPartitions.add(trainPriorPartitions.get(fold))
	testReadPartitions.add(testPriorPartitions.get(fold))

	Partition testLabelPartition = trustsPartitions.get(fold)


	// training
	Database trainDB = data.getDatabase(trainWritePartitions.get(fold), (Partition []) trainReadPartitions.toArray())
	Set<GroundAtom> allTrusts = Queries.getAllAtoms(trainDB, trusts)
	double sum = 0.0;
	for (GroundAtom atom : allTrusts)
		sum += atom.getValue()
	trainDB.close()
	data.getInserter(prior, trainPriorPartitions.get(fold)).insertValue(sum / allTrusts.size(), constant)
	log.info("Computed training prior for fold {} of {}", fold, sum / allTrusts.size())

	// testing
	Database testDB = data.getDatabase(testWritePartitions.get(fold), (Partition []) testReadPartitions.toArray())
	allTrusts = Queries.getAllAtoms(testDB, trusts)
	sum = 0.0;
	for (GroundAtom atom : allTrusts)
		sum += atom.getValue()
	testDB.close()
	data.getInserter(prior, testPriorPartitions.get(fold)).insertValue(sum / allTrusts.size(), constant)
	log.info("Computed testing prior for fold {} of {}", fold, sum / allTrusts.size())

	// reopen databases
	trainDB = data.getDatabase(trainWritePartitions.get(fold), (Partition []) trainReadPartitions.toArray())
	testDB = data.getDatabase(testWritePartitions.get(fold), (Partition []) testReadPartitions.toArray())


	/*
	 * POPULATE TRAINING DATABASE
	 * Get all knows pairs, 
	 */
	int rv = 0, ob = 0
	ResultList allGroundings = trainDB.executeQuery(Queries.getQueryForAllAtoms(knows))
	for (int i = 0; i < allGroundings.size(); i++) {
		GroundTerm [] grounding = allGroundings.get(i)
		GroundAtom atom = trainDB.getAtom(trusts, grounding)
		if (atom instanceof RandomVariableAtom) {
			rv++
			trainDB.commit((RandomVariableAtom) atom);
		} else
			ob++
	}
	System.out.println("Saw " + rv + " rvs and " + ob + " obs")

	/*
	 * POPULATE TEST DATABASE
	 */
	allGroundings = testDB.executeQuery(Queries.getQueryForAllAtoms(knows))
	for (int i = 0; i < allGroundings.size(); i++) {
		GroundTerm [] grounding = allGroundings.get(i)
		GroundAtom atom = testDB.getAtom(trusts, grounding)
		if (atom instanceof RandomVariableAtom) {
			testDB.commit((RandomVariableAtom) atom);
		}
	}

	testDB.close()

	Partition dummy = new Partition(99999)
	Partition dummy2 = new Partition(19999)
	Database labelsDB = data.getDatabase(dummy, [trusts] as Set, trainLabelPartition)

	for (Map.Entry<String, ConfigBundle> method : methodConfigs.entrySet()) {
		for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
			k.setWeight(weights.get(k))

		/*
		 * Weight learning
		 */
		learn(m, trainDB, labelsDB, method.getValue(), log)

		System.out.println("Learned model " + method.getKey() + "\n" + m.toString())

		/*
		 * Inference on test set
		 */
		testDB = data.getDatabase(testWritePartitions.get(fold), (Partition []) testReadPartitions.toArray())
		for (int i = 0; i < allGroundings.size(); i++) {
			GroundTerm [] grounding = allGroundings.get(i)
			GroundAtom atom = testDB.getAtom(trusts, grounding)
			if (atom instanceof RandomVariableAtom) {
				atom.setValue(0.0)
			}
		}
		MPEInference mpe = new MPEInference(m, testDB, baseConfig)
		FullInferenceResult result = mpe.mpeInference()
		testDB.close()

		/*
		 * Evaluation
		 * implement area under PR curve
		 */
		Database resultsDB = data.getDatabase(dummy2, testWritePartitions.get(fold))
		def comparator = new SimpleRankingComparator(resultsDB)
		def groundTruthDB = data.getDatabase(testLabelPartition, [trusts] as Set)
		comparator.setBaseline(groundTruthDB)


		def metrics = [RankingScore.AUPRC, RankingScore.NegAUPRC, RankingScore.AreaROC]
		double [] score = new double[metrics.size()]

		for (int i = 0; i < metrics.size(); i++) {
			comparator.setRankingScore(metrics.get(i))
			score[i] = comparator.compare(trusts)
		}
		System.out.println("Area under positive-class PR curve: " + score[0])
		System.out.println("Area under negative-class PR curve: " + score[1])
		System.out.println("Area under ROC curve: " + score[2])

		results.get(method.getKey()).add(fold, score)
		resultsDB.close()
		groundTruthDB.close()
	}
	trainDB.close()
	labelsDB.close()
}

for (String method : methodConfigs.keySet()) {
	def methodStats = results.get(method)
	sum = new double[3];
	sumSq = new double[3];
	for (int fold = 0; fold < folds; fold++) {
		def score = methodStats.get(fold)
		for (int i = 0; i < 3; i++) {
			sum[i] += score[i];
			sumSq[i] += score[i] * score[i];
		}
		System.out.println("Method " + method + ", fold " + fold +", auprc positive: "
				+ score[0] + ", negative: " + score[1] + ", auROC: " + score[2])
	}

	mean = new double[3];
	variance = new double[3];
	for (int i = 0; i < 3; i++) {
		mean[i] = sum[i] / folds;
		variance[i] = sumSq[i] / folds - mean[i] * mean[i];
	}

	System.out.println();
	System.out.println("Method " + method + ", auprc positive: (mean/variance) "
			+ mean[0] + "  /  " + variance[0] );
	System.out.println("Method " + method + ", auprc negative: (mean/variance) "
			+ mean[1] + "  /  " + variance[1] );
	System.out.println("Method " + method + ", auROC: (mean/variance) "
			+ mean[2] + "  /  " + variance[2] );
	System.out.println();
}


private void learn(Model m, Database db, Database labelsDB, ConfigBundle config, Logger log) {
	switch(config.getString("method", "")) {
		case "MLE":
			MaxLikelihoodMPE mle = new MaxLikelihoodMPE(m, db, labelsDB, config)
			mle.learn()
			break
		case "MPLE":
			MaxPseudoLikelihood mple = new MaxPseudoLikelihood(m, db, labelsDB, config)
			mple.learn()
			break
		case "MM":
			MaxMargin mm = new MaxMargin(m, db, labelsDB, config)
			mm.learn()
			break
		case "HEMRandOM":
			HardEMRandOM hardRandOM = new HardEMRandOM(m, db, labelsDB, config)
			hardRandOM.setSlackPenalty(10000)
		//hardRandOM.learn()
			break
		case "FirstOrderRandOM":
			FirstOrderMetropolisRandOM randOM = new FirstOrderMetropolisRandOM(m, db, labelsDB, config)
			randOM.learn()
			break
		case "IncompatibilityRandOM":
			IncompatibilityMetropolisRandOM randOM = new IncompatibilityMetropolisRandOM(m, db, labelsDB, config);
			randOM.learn();
			break
		default:
			throw new IllegalArgumentException("Unrecognized method.");
	}
}