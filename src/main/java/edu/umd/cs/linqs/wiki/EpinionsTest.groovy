package edu.umd.cs.linqs.wiki

import java.util.Set;

import junit.framework.TestResult;

import org.eclipse.jdt.internal.core.util.LRUCache.Stats;
import org.slf4j.Logger
import org.slf4j.LoggerFactory

import edu.emory.mathcs.utils.ConcurrencyUtils
import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood
import edu.umd.cs.psl.application.learning.weight.random.FirstOrderMetropolisRandOM
import edu.umd.cs.psl.application.learning.weight.random.HardEMRandOM
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.core.*
import edu.umd.cs.psl.core.inference.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.database.DatabasePopulator
import edu.umd.cs.psl.database.DatabaseQuery
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.ResultList
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.*
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionComparator
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionStatistics
import edu.umd.cs.psl.evaluation.statistics.filter.MaxValueFilter
import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.Model;
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.GroundTerm
import edu.umd.cs.psl.model.argument.UniqueID
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.atom.GroundAtom
import edu.umd.cs.psl.model.atom.QueryAtom
import edu.umd.cs.psl.model.atom.RandomVariableAtom;
import edu.umd.cs.psl.model.function.AttributeSimilarityFunction
import edu.umd.cs.psl.ui.loading.*
import edu.umd.cs.psl.util.database.Queries
import edu.umd.cs.psl.model.kernel.CompatibilityKernel;
import edu.umd.cs.psl.model.parameters.PositiveWeight;
import edu.umd.cs.psl.model.parameters.Weight
import edu.umd.cs.psl.model.predicate.Predicate

import com.google.common.collect.Iterables;

//methods = ["RandOM", "MM1", "MM10", "MM100", "MM1000", "MLE"]
//methods = ["MM1", "MM10", "MM100", "MM1000", "MLE"]
//methods = ["None"]
methods = ["NONE", "MLE", "MPLE", "MM1", "MM10"]

Logger log = LoggerFactory.getLogger(this.class)

ConfigManager cm = ConfigManager.getManager()
ConfigBundle epinionsBundle = cm.getBundle("wiki")

def defaultPath = System.getProperty("java.io.tmpdir")
String dbpath = epinionsBundle.getString("dbpath", defaultPath + "psl")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), epinionsBundle)

PSLModel m = new PSLModel(this, data)

/*
 * DEFINE MODEL
 */
m.add predicate: "knows", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "trusts", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

//prior
m.add rule : ~(trusts(A,B)), weight: 1.0

m.add rule: (knows(A,B) & knows(B,C) & knows(A,C) & trusts(A,B) & trusts(B,C)) >> trusts(A,C), weight: 1.0   //FFpp
m.add rule: (knows(A,B) & knows(B,C) & knows(A,C) & trusts(A,B) & ~trusts(B,C)) >> ~trusts(A,C), weight: 1.0 //FFpm
m.add rule: (knows(A,B) & knows(B,C) & knows(A,C) & ~trusts(A,B) & trusts(B,C)) >> ~trusts(A,C), weight: 1.0 //FFmp
m.add rule: (knows(A,B) & knows(B,C) & knows(A,C) & ~trusts(A,B) & ~trusts(B,C)) >> trusts(A,C), weight: 1.0 //FFmm

m.add rule: (knows(A,B) & knows(C,B) & knows(A,C) & trusts(A,B) & trusts(C,B)) >> trusts(A,C), weight: 1.0   //FBpp
m.add rule: (knows(A,B) & knows(C,B) & knows(A,C) & trusts(A,B) & ~trusts(C,B)) >> ~trusts(A,C), weight: 1.0 //FBpm
m.add rule: (knows(A,B) & knows(C,B) & knows(A,C) & ~trusts(A,B) & trusts(C,B)) >> ~trusts(A,C), weight: 1.0 //FBmp
m.add rule: (knows(A,B) & knows(C,B) & knows(A,C) & ~trusts(A,B) & ~trusts(C,B)) >> trusts(A,C), weight: 1.0 //FBmm

m.add rule: (knows(B,A) & knows(B,C) & knows(A,C) & trusts(B,A) & trusts(B,C)) >> trusts(A,C), weight: 1.0   //BFpp
m.add rule: (knows(B,A) & knows(B,C) & knows(A,C) & trusts(B,A) & ~trusts(B,C)) >> ~trusts(A,C), weight: 1.0 //BFpm
m.add rule: (knows(B,A) & knows(B,C) & knows(A,C) & ~trusts(B,A) & trusts(B,C)) >> ~trusts(A,C), weight: 1.0 //BFmp
m.add rule: (knows(B,A) & knows(B,C) & knows(A,C) & ~trusts(B,A) & ~trusts(B,C)) >> trusts(A,C), weight: 1.0 //BFmm

m.add rule: (knows(B,A) & knows(C,B) & knows(A,C) & trusts(B,A) & trusts(C,B)) >> trusts(A,C), weight: 1.0   //BBpp
m.add rule: (knows(B,A) & knows(C,B) & knows(A,C) & trusts(B,A) & ~trusts(C,B)) >> ~trusts(A,C), weight: 1.0 //BBpm
m.add rule: (knows(B,A) & knows(C,B) & knows(A,C) & ~trusts(B,A) & trusts(C,B)) >> ~trusts(A,C), weight: 1.0 //BBmp
m.add rule: (knows(B,A) & knows(C,B) & knows(A,C) & ~trusts(B,A) & ~trusts(C,B)) >> trusts(A,C), weight: 1.0 //BBmm


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
// ratio of train to test splits
trainTestRatio = 0.5
// ratio of documents to keep (throw away the rest)
filterRatio = 1.0

List<Partition> trustsPartitions = new ArrayList<Partition>(folds)
List<Partition> knowsPartitions = new ArrayList<Partition>(folds)
List<Partition> trainWritePartitions = new ArrayList<Partition>(folds)
List<Partition> testWritePartitions = new ArrayList<Partition>(folds)

Random rand = new Random(0)
double trainingObservedRatio = 0.6

for (int i = 0; i < folds; i++) {
	knowsPartitions.add(i, new Partition(i + 2))
	trustsPartitions.add(i, new Partition(i + folds + 2))
	trainWritePartitions.add(i, new Partition(i + 2*folds + 2))
	testWritePartitions.add(i, new Partition(i + 3*folds + 2))
}

List<Set<GroundingWrapper>> groundings = FoldUtils.splitGroundings(data, [trusts, knows], [fullTrusts, fullKnows], folds)
for (int i = 0; i < folds; i++) {
	FoldUtils.copy(data, fullKnows, knowsPartitions.get(i), knows, groundings.get(i))
	FoldUtils.copy(data, fullTrusts, trustsPartitions.get(i), trusts, groundings.get(i))
}


Map<String, List<DiscretePredictionStatistics>> results = new HashMap<String, List<DiscretePredictionStatistics>>()
for (String method : methods)
	results.put(method, new ArrayList<DiscretePredictionStatistics>())

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

	Partition testLabelPartition = trustsPartitions.get(fold)

	// open databases
	Database trainDB = data.getDatabase(trainWritePartitions.get(fold), (Partition []) trainReadPartitions.toArray())
	Database testDB = data.getDatabase(testWritePartitions.get(fold), (Partition []) testReadPartitions.toArray())


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

	/**
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

	Partition dummy = new Partition(99999)
	Database labelsDB = data.getDatabase(dummy, [trusts] as Set, trainLabelPartition)

	// get all default weights
	Map<CompatibilityKernel,Weight> weights = new HashMap<CompatibilityKernel, Weight>()
	for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
		weights.put(k, k.getWeight());

	for (String method : methods) {
		for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
			k.setWeight(weights.get(k))

		/*
		 * Weight learning
		 */
		learn(m, trainDB, labelsDB, epinionsBundle, method, log)

		System.out.println("Learned model " + method + "\n" + m.toString())

		/*
		 * Inference on test set
		 */
		MPEInference mpe = new MPEInference(m, testDB, epinionsBundle)
		FullInferenceResult result = mpe.mpeInference()
		System.out.println("Objective: " + result.getTotalWeightedIncompatibility())

		/*
		 * Evaluation
		 * TODO: implement area under PR curve
		 */
		def comparator = new DiscretePredictionComparator(testDB)
		def groundTruthDB = data.getDatabase(testLabelPartition, [trusts] as Set)
		comparator.setBaseline(groundTruthDB)
		comparator.setThreshold(0.5)

		//System.out.println("totalTestExamples " + totalTestExamples)
		DiscretePredictionStatistics stats = comparator.compare(trusts)
		System.out.println("F1 score " + stats.getF1(
				DiscretePredictionStatistics.BinaryClass.NEGATIVE))

		results.get(method).add(fold, stats)

		groundTruthDB.close()
	}
	trainDB.close()
}

for (String method : methods) {
	def methodStats = results.get(method)
	for (int fold = 0; fold < folds; fold++) {
		def stats = methodStats.get(fold)
		def b = DiscretePredictionStatistics.BinaryClass.POSITIVE
		System.out.println("Method " + method + ", fold " + fold +", acc " + stats.getAccuracy() +
				", prec " + stats.getPrecision(b) + ", rec " + stats.getRecall(b) +
				", F1 " + stats.getF1(b))
	}
}


private void learn(Model m, Database db, Database labelsDB, ConfigBundle config, String method, Logger log) {
	switch(method) {
		case "MLE":
			MaxLikelihoodMPE mle = new MaxLikelihoodMPE(m, db, labelsDB, config)
			mle.learn()
			break
		case "MPLE":
			MaxPseudoLikelihood mple = new MaxPseudoLikelihood(m, db, labelsDB, config)
			mple.learn()
			break
		case "MM0.1":
			MaxMargin mm = new MaxMargin(m, db, labelsDB, config)
			mm.setSlackPenalty(0.1)
			mm.learn()
			break
		case "MM1":
			MaxMargin mm = new MaxMargin(m, db, labelsDB, config)
			mm.setSlackPenalty(1)
			mm.learn()
			break
		case "MM10":
			MaxMargin mm = new MaxMargin(m, db, labelsDB, config)
			mm.setSlackPenalty(10)
			mm.learn()
			break
		case "MM100":
			MaxMargin mm = new MaxMargin(m, db, labelsDB, config)
			mm.setSlackPenalty(100)
			mm.learn()
			break
		case "MM1000":
			MaxMargin mm = new MaxMargin(m, db, labelsDB, config)
			mm.setSlackPenalty(1000)
			mm.learn()
			break
		case "HEMRandOM":
			HardEMRandOM hardRandOM = new HardEMRandOM(m, db, labelsDB, config)
			hardRandOM.setSlackPenalty(10000)
		//hardRandOM.learn()
			break
		case "RandOM":
			FirstOrderMetropolisRandOM randOM = new FirstOrderMetropolisRandOM(m, db, labelsDB, config)
			randOM.learn()
			break
		default:
			log.error("Invalid method ")
	}
}