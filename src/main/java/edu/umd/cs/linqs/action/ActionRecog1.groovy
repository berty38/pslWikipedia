package edu.umd.cs.linqs.action

import org.slf4j.Logger
import org.slf4j.LoggerFactory

import com.google.common.collect.Iterables

import edu.umd.cs.linqs.WeightLearner
import edu.umd.cs.linqs.wiki.ExperimentConfigGenerator
import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.LossBalancingType
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.NormScalingType
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.database.DatabasePopulator
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.loading.Inserter
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.FullInferenceResult
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionComparator
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionStatistics
import edu.umd.cs.psl.evaluation.statistics.filter.MaxValueFilter
import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.GroundTerm
import edu.umd.cs.psl.model.argument.IntegerAttribute
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.atom.GroundAtom
import edu.umd.cs.psl.model.atom.QueryAtom
import edu.umd.cs.psl.model.atom.RandomVariableAtom
import edu.umd.cs.psl.model.kernel.CompatibilityKernel
import edu.umd.cs.psl.model.parameters.Weight
import edu.umd.cs.psl.ui.loading.*
import edu.umd.cs.psl.util.database.Queries


/*** CONFIGURATION PARAMETERS ***/

def dataPath = "./data/action/"
def filePfx = dataPath + "d1_";

Logger log = LoggerFactory.getLogger(this.class)
ConfigManager cm = ConfigManager.getManager();
ConfigBundle cb = cm.getBundle("action");

def defPath = System.getProperty("java.io.tmpdir") + "/action"
def dbpath = cb.getString("dbpath", defPath)
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), cb)
folds = 44

def sq = cb.getBoolean("squared", true);

ExperimentConfigGenerator configGenerator = new ExperimentConfigGenerator("action");

/*
 * SET MODEL TYPES
 *
 * Options:
 * "quad" HLEF
 * "bool" MLN
 */
configGenerator.setModelTypes(["quad"]);

/*
 * SET LEARNING ALGORITHMS
 *
 * Options:
 * "MLE" (MaxLikelihoodMPE)
 * "MPLE" (MaxPseudoLikelihood)
 * "MM" (MaxMargin)
 */
//methods = ["MLE","MPLE","MM"];
methods = ["MLE"];
configGenerator.setLearningMethods(methods);

/* MLE/MPLE options */
configGenerator.setVotedPerceptronStepCounts([100]);
configGenerator.setVotedPerceptronStepSizes([(double) 1.0]);

/* MM options */
//configGenerator.setMaxMarginSlackPenalties([(double) 0.1, (double) 0.5, (double) 1.0]);
configGenerator.setMaxMarginSlackPenalties([(double) 0.1]);
configGenerator.setMaxMarginLossBalancingTypes([LossBalancingType.NONE]);
configGenerator.setMaxMarginNormScalingTypes([NormScalingType.NONE]);
configGenerator.setMaxMarginSquaredSlackValues([false]);

List<ConfigBundle> configs = configGenerator.getConfigs();
for (ConfigBundle config : configs)
	System.out.println(config);


/*** MODEL DEFINITION ***/

log.info("Initializing model ...");

PSLModel m = new PSLModel(this, data);

/* CONSTANTS */

actions = [1,2,3,4,5];
actionNames = ["crossing","standing","queueing","walking","talking"];

/* PREDICATES */

// types (might not need some of these)
//m.add predicate: "frame", types: [ArgumentType.UniqueID];
//m.add predicate: "bbox", types: [ArgumentType.UniqueID];
//m.add predicate: "action", types: [ArgumentType.UniqueID];

// target
m.add predicate: "doing", types: [ArgumentType.UniqueID,ArgumentType.Integer];
m.add predicate: "sameObj", types: [ArgumentType.UniqueID,ArgumentType.UniqueID];

// observed
m.add predicate: "inFrame", types: [ArgumentType.UniqueID,ArgumentType.Integer,ArgumentType.Integer];
m.add predicate: "inSameFrame", types: [ArgumentType.UniqueID,ArgumentType.UniqueID];
m.add predicate: "inSeqFrames", types: [ArgumentType.UniqueID,ArgumentType.UniqueID];
m.add predicate: "dims", types: [ArgumentType.UniqueID,ArgumentType.Integer,ArgumentType.Integer,ArgumentType.Integer,ArgumentType.Integer];
m.add predicate: "hogAction", types: [ArgumentType.UniqueID,ArgumentType.Integer];
m.add predicate: "acdAction", types: [ArgumentType.UniqueID,ArgumentType.Integer];
//m.add predicate: "seqFrames", types: [ArgumentType.Integer,ArgumentType.Integer];

// derived
//m.add function: "seqFrames", implementation: new SequentialTest();
m.add function: "far", implementation: new DistanceFunction();

/* ACTION RULES */

for (int a1 : actions) {
	// HOG-based SVM probabilities
	m.add rule: hogAction(BB,a1) >> doing(BB,a1), weight: 1.0, squared: sq;
	// ACD-based SVM probabilities
	m.add rule: acdAction(BB,a1) >> doing(BB,a1), weight: 1.0, squared: sq;
	// Relational rules
	for (int a2 : actions) {
		// If BB1,BB2 in same frame, and BB1 is doing action a1, and BB2 is close, then BB2 is doing action a2.
		m.add rule: ( inSameFrame(BB1,BB2) & doing(BB1,a1) & dims(BB1,X1,Y1,W1,H1) & dims(BB2,X2,Y2,W2,H2) & ~far(X1,X2,Y1,Y2,W1,W2,H1,H2) ) >> doing(BB2,a2), weight: 1.0, squared: sq;
		// If BB1,BB2 in same frame, and BB1 is doing action a1, and BB2 is far, then BB2 is doing action a2.
		m.add rule: ( inSameFrame(BB1,BB2) & doing(BB1,a1) & dims(BB1,X1,Y1,W1,H1) & dims(BB2,X2,Y2,W2,H2) & far(X1,X2,Y1,Y2,W1,W2,H1,H2) ) >> doing(BB2,a2), weight: 1.0, squared: sq;
		// If BB1,BB2 in sequential frames, and BB1 is doing action a1, then BB2 is doing action a2.
		m.add rule: ( sameObj(BB1,BB2) & doing(BB1,a1) ) >> doing(BB2,a2), weight: 1.0, squared: sq;
	}
	// Priors on actions
	m.add rule: ~doing(BB1,a1), weight: 1.0, squared: sq;
}
// Functional constraint on doing means that it should sum to 1 for each BB
m.add PredicateConstraint.Functional, on: doing;

/* ID MAINTENANCE: IN-FRAME RULES */

// If BB1,BB2 in same frame, cannot be same object.
m.add rule: inSameFrame(BB1,BB2) >> ~sameObj(BB1,BB2), constraint: true;

/* ID MAINTENANCE: BETWEEN-FRAME RULES */

// If BB1 in F1, BB2 in F2, and F1,F2 are sequential, and BB1,BB2 are NEAR, then BB1,BB2 are same object.
m.add rule: ( inSeqFrames(BB1,BB2) & dims(BB1,X1,Y1,W1,H1) & dims(BB2,X2,Y2,W2,H2) & ~far(X1,X2,Y1,Y2,W1,W2,H1,H2) ) >> sameObj(BB1,BB2), weight: 1.0, squared: sq;
//m.add rule: ( inFrame(BB1,S1,F1) & inFrame(BB2,S2,F2) & seqFrames(F1,F2) 
//			& dims(BB1,X1,Y1,W1,H1) & dims(BB2,X2,Y2,W2,H2) & ~far(X1,X2,Y1,Y2,W1,W2,H1,H2) ) >> sameObj(BB1,BB2), weight: 1.0, squared: sq;

// Prior on sameObj
m.add rule: ~sameObj(BB1,BB2), weight: 1.0, squared: sq;

log.info("Model: {}", m)

/* get all default weights */
Map<CompatibilityKernel,Weight> initWeights = new HashMap<CompatibilityKernel, Weight>();
for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
	initWeights.put(k, k.getWeight());

/*** LOAD DATA ***/

log.info("Loading data ...");

/* Create obs/label partitions for each sequence */
int partCnt = 0;
Partition[][] partitions = new Partition[2][folds];
for (int fold = 0; fold < folds; fold++) {
	partitions[0][fold] = new Partition(partCnt++);	// observations
	partitions[1][fold] = new Partition(partCnt++);	// labels
}

Inserter[] inserters;

/* Ground truth */
inserters = InserterUtils.getMultiPartitionInserters(data, doing, partitions[1], folds);
InserterUtils.loadDelimitedDataMultiPartition(inserters, filePfx + "action.txt");
inserters = InserterUtils.getMultiPartitionInserters(data, sameObj, partitions[1], folds);
InserterUtils.loadDelimitedDataMultiPartition(inserters, filePfx + "sameObj.txt");

/* Observations */
inserters = InserterUtils.getMultiPartitionInserters(data, inFrame, partitions[0], folds);
InserterUtils.loadDelimitedDataMultiPartition(inserters, filePfx + "inframe.txt");
inserters = InserterUtils.getMultiPartitionInserters(data, inSameFrame, partitions[0], folds);
InserterUtils.loadDelimitedDataMultiPartition(inserters, filePfx + "insameframe.txt");
inserters = InserterUtils.getMultiPartitionInserters(data, dims, partitions[0], folds);
InserterUtils.loadDelimitedDataMultiPartition(inserters, filePfx + "coords.txt");
inserters = InserterUtils.getMultiPartitionInserters(data, hogAction, partitions[0], folds);
InserterUtils.loadDelimitedDataTruthMultiPartition(inserters, filePfx + "hogaction.txt");
inserters = InserterUtils.getMultiPartitionInserters(data, acdAction, partitions[0], folds);
InserterUtils.loadDelimitedDataTruthMultiPartition(inserters, filePfx + "acdaction.txt");
inserters = InserterUtils.getMultiPartitionInserters(data, inSeqFrames, partitions[0], folds);
InserterUtils.loadDelimitedDataMultiPartition(inserters, filePfx + "seqframes.txt");
//inserters = InserterUtils.getMultiPartitionInserters(data, seqFrames, partitions[0], folds);
//InserterUtils.loadDelimitedDataMultiPartition(inserters, filePfx + "seqframes.txt", "\t", 1000);


/** GLOBAL DATA FOR DB POPULATION **/

List<HashMap<Integer,List<GroundTerm>>> seqs = new ArrayList<HashMap<Integer,List<GroundTerm>>>();
for (int s = 0; s < folds; s++) {
	seqs.add(new HashMap<Integer,List<GroundTerm>>());
}
def toClose = [inFrame, inSameFrame, dims, hogAction, acdAction, inSeqFrames] as Set;
Database db = data.getDatabase(new Partition(partCnt++), toClose, partitions[0]);
Set<GroundAtom> atoms = Queries.getAllAtoms(db, inFrame);
for (GroundAtom a : atoms) {
	// Get terms
	GroundTerm[] terms = a.getArguments();
	GroundTerm bbox = terms[0];
	int s = ((IntegerAttribute)terms[1]).getValue().intValue() - 1;
	int f = ((IntegerAttribute)terms[2]).getValue().intValue();
	Map<Integer,List<GroundTerm>> frames = seqs[s];
	List<GroundTerm> bboxes;
	if (frames.get(f) != null)
		bboxes = frames.get(f);
	else {
		bboxes = new ArrayList<GroundTerm>();
		frames.put(f, bboxes);
	}
	bboxes.add(bbox);
}
db.close();
Set<GroundTerm> actionTerms = new HashSet<GroundTerm>();
for (int a : actions) {
	actionTerms.add(new IntegerAttribute(a));
}


/*** RUN EXPERIMENTS ***/

log.info("Starting experiments.");

Map<String, List<DiscretePredictionStatistics>> stats_doing = new HashMap<String, List<DiscretePredictionStatistics>>()
for (ConfigBundle method : configs)
	stats_doing.put(method, new ArrayList<DiscretePredictionStatistics>())
Map<String, List<DiscretePredictionStatistics>> stats_sameObj = new HashMap<String, List<DiscretePredictionStatistics>>()
for (ConfigBundle method : configs)
	stats_sameObj.put(method, new ArrayList<DiscretePredictionStatistics>())
	
//for (int fold = 0; fold < folds; fold++) {
for (int fold = 0; fold < 1; fold++) {

	/** SPLIT DATA **/
	
	log.info("Splitting data ...");
	
	/* To construct training set: query for all of the atoms from each scene, except for hold-out. */
	List<Partition> trainPartsObs = new ArrayList<Partition>();
	List<Partition> trainPartsLab = new ArrayList<Partition>();
	for (int s = 0; s < folds; s++) {
		if (s == fold)
			continue;
		trainPartsObs.add(partitions[0][s]);
		trainPartsLab.add(partitions[1][s]);
	}
	testPartObs = partitions[0][fold];
	testPartLab = partitions[1][fold];
	
	/** POPULATE DB ***/

	log.info("Populating databases ...");
		
	Partition write_tr = new Partition(partCnt++);
	Partition write_te = new Partition(partCnt++);
	Database trainDB = data.getDatabase(write_tr, toClose, (Partition[])trainPartsObs.toArray());
	Database testDB = data.getDatabase(write_te, toClose, testPartObs);

	/* Populate doing predicate. */
	Variable BBox = new Variable("BBox");
	Variable Action = new Variable("Action");
	Map<Variable, Set<GroundTerm>> subs = new HashMap<Variable, Set<GroundTerm>>();
	subs.put(Action, actionTerms);
	// Get all bbox ground terms
	Set<GroundTerm> bboxTerms_tr = new HashSet<GroundTerm>();
	Set<GroundTerm> bboxTerms_te = new HashSet<GroundTerm>();
	for (int s = 0; s < folds; s++) {
		Set<GroundTerm> curSet = (s != fold) ? bboxTerms_tr : bboxTerms_te;
		Map<Integer,ArrayList<GroundTerm>> frames = seqs[s];
		for (ArrayList<GroundTerm> f : frames.values()) {
			curSet.addAll(f);
		}
	}
	// Training
	subs.put(BBox, bboxTerms_tr);
	DatabasePopulator dbPop = new DatabasePopulator(trainDB);
	dbPop.populate(new QueryAtom(doing, BBox, Action), subs);
	// Testing
	subs.put(BBox, bboxTerms_te);
	dbPop = new DatabasePopulator(testDB);
	dbPop.populate(new QueryAtom(doing, BBox, Action), subs);
	int numTestEx_doing = bboxTerms_te.size() * actions.size();
	
	/* Populate sameObj predicate.*/
	int numTestEx_sameObj = 0;
	for (int s = 0; s < folds; s++) {
		HashMap<Integer,List<GroundTerm>> frames = seqs[s-1];
		List<Integer> frameID = frames.keySet().asList();
		Collections.sort(frameID);
		List<GroundTerm> bboxes1, bboxes2;
		for (int i = 0; i < frameID.size()-1; i++) {
			bboxes1 = frames.get(frameID[i]);
			bboxes2 = frames.get(frameID[i+1]);
			for (GroundTerm bb1 : bboxes1) {
				for (GroundTerm bb2 : bboxes2) {
					if (s != fold) {
						RandomVariableAtom rv = trainDB.getAtom(sameObj, bb1, bb2);
						trainDB.commit(rv);
					}
					else {
						RandomVariableAtom rv = testDB.getAtom(sameObj, bb1, bb2);
						testDB.commit(rv);
						++numTestEx_sameObj;
					}
				}
			}
		}
	}
	

	/* Need to close testDB so that we can use write_te for multiple databases. */	
	testDB.close();
	
	/* Label DBs */
	Database labelDB = data.getDatabase(new Partition(partCnt++), [doing,sameObj] as Set, (Partition[])trainPartsLab.toArray());

	/*** EXPERIMENT ***/
	
	log.info("Starting experiment ...");
	
	for (int configIndex = 0; configIndex < configs.size(); configIndex++) {
		ConfigBundle config = configs.get(configIndex);
		def configName = config.getString("name", "");
		def method = config.getString("learningmethod", "");

		/* Weight learning */
		WeightLearner.learn(method, m, trainDB, labelDB, initWeights, config, log)

		log.info("Learned model {}: \n {}", configName, m.toString())

		/* Inference on test set */
		testDB = data.getDatabase(write_te, toClose, testPartObs);
		Set<GroundAtom> targetAtoms = Queries.getAllAtoms(testDB, doing)
		for (RandomVariableAtom rv : Iterables.filter(targetAtoms, RandomVariableAtom))
			rv.setValue(0.0)
		targetAtoms = Queries.getAllAtoms(testDB, sameObj);
		for (RandomVariableAtom rv : Iterables.filter(targetAtoms, RandomVariableAtom))
			rv.setValue(0.0)
		MPEInference mpe = new MPEInference(m, testDB, config)
		FullInferenceResult result = mpe.mpeInference()
		log.info("Objective: {}", result.getTotalWeightedIncompatibility())
		testDB.close();
	
		/* Evaluate doing predicate */
		Database predDB = data.getDatabase(write_te, [doing] as Set);
		Database truthDB = data.getDatabase(testPartLab, [doing] as Set);
		def comparator = new DiscretePredictionComparator(predDB);
		comparator.setBaseline(truthDB);
		comparator.setResultFilter(new MaxValueFilter(doing, 1));
		comparator.setThreshold(Double.MIN_VALUE) // treat best value as true as long as it is nonzero
		DiscretePredictionStatistics stats = comparator.compare(doing, numTestEx_doing);
		System.out.println("F1 Action:  " + stats.getF1(DiscretePredictionStatistics.BinaryClass.POSITIVE));
		stats_doing.get(config).add(fold, stats)
		predDB.close();
		truthDB.close();
	
		/* Evaluate doing predicate */
		predDB = data.getDatabase(write_te, [doing] as Set);
		truthDB = data.getDatabase(testPartLab, [doing] as Set);
		comparator = new DiscretePredictionComparator(predDB);
		comparator.setBaseline(truthDB);
		comparator.setThreshold(0.5);
		stats = comparator.compare(doing, numTestEx_sameObj);
		System.out.println("F1 SameObj:  " + stats.getF1(DiscretePredictionStatistics.BinaryClass.POSITIVE));
		stats_sameObj.get(config).add(fold, stats)
		predDB.close();
		truthDB.close();
	}
	
	/* Close all databases. */
	trainDB.close();
	labelDB.close();
}

//log.warn("\n\nRESULTS\n");
//for (ConfigBundle config : configs) {
//	def configName = config.getString("name", "")
//	def scores = stats_doing.get(config);
//	for (int fold = 0; fold < folds; fold++) {
//		def score = scores.get(fold)
//		log.warn("{} \t{}\t{}\t{}", configName, fold, score[0], score[1]);
//	}
//}
