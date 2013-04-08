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
import edu.umd.cs.psl.database.ResultList
import edu.umd.cs.psl.database.loading.Inserter
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.FullInferenceResult
import edu.umd.cs.psl.evaluation.statistics.ContinuousPredictionComparator
import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.GroundTerm
import edu.umd.cs.psl.model.argument.IntegerAttribute
import edu.umd.cs.psl.model.argument.UniqueID
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.atom.GroundAtom
import edu.umd.cs.psl.model.atom.QueryAtom
import edu.umd.cs.psl.model.atom.RandomVariableAtom
import edu.umd.cs.psl.model.formula.Formula;
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
m.add predicate: "dims", types: [ArgumentType.UniqueID,ArgumentType.Integer,ArgumentType.Integer,ArgumentType.Integer,ArgumentType.Integer];
m.add predicate: "hogAction", types: [ArgumentType.UniqueID,ArgumentType.Integer];
m.add predicate: "acdAction", types: [ArgumentType.UniqueID,ArgumentType.Integer];

// derived
m.add function: "seqFrames", implementation: new SequentialTest();
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

/* ID MAINTENANCE: IN-FRAME RULES */

// If BB1,BB2 in same frame, cannot be same object.
m.add rule: inSameFrame(BB1,BB2) >> ~sameObj(BB1,BB2), constraint: true;

/* ID MAINTENANCE: BETWEEN-FRAME RULES */

// If BB1 in F1, BB2 in F2, and F1,F2 are sequential, and BB1,BB2 are NEAR, then BB1,BB2 are same object.
m.add rule: ( inFrame(BB1,S1,F1) & inFrame(BB2,S2,F2) & seqFrames(F1,F2) 
			& dims(BB1,X1,Y1,W1,H1) & dims(BB2,X2,Y2,W2,H2) & ~far(X1,X2,Y1,Y2,W1,W2,H1,H2) ) >> sameObj(BB1,BB2), weight: 1.0, squared: sq;

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
Partition[][] partitions = new Partition[2][folds];
for (int fold = 0; fold < folds; fold++) {
	partitions[0][fold] = data.getNextPartition();	// observations
	partitions[1][fold] = data.getNextPartition();	// labels
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


/*** RUN EXPERIMENTS ***/

Map<ConfigBundle,ArrayList<Double>> expResults = new HashMap<String,ArrayList<Double>>();
for (ConfigBundle config : configs) {
	expResults.put(config, new ArrayList<Double>(folds));
}

//for (int fold = 0; fold < folds; fold++) {
for (int fold = 0; fold < 1; fold++) {

	/** SPLIT DATA **/
	
	Partition write_tr = data.getNextPartition();
	Partition write_te = data.getNextPartition();
	
	def toClose = [inFrame, inSameFrame, dims, hogAction, acdAction] as Set;
	
	/* To construct training set: query for all of the atoms from each scene, except for hold-out. */
	ArrayList<Partition> trainPartsObs = new ArrayList<Partition>();
	ArrayList<Partition> trainPartsLab = new ArrayList<Partition>();
	for (int s = 0; s < folds; s++) {
		if (s == fold)
			continue;
		trainPartsObs.add(partitions[0][s]);
		trainPartsObs.add(partitions[1][s]);
	}	
	
	/** POPULATE TRAIN DB ***/

	/* Populate doing predicate. */
	Database trainDB = data.getDatabase(write_tr, toClose, (Partition[])trainPartsObs.toArray());
	Variable BBox = new Variable("BBox");
	Variable Action = new Variable("Action");
	Set<GroundAtom> atoms = Queries.getAllAtoms(trainDB, dims);
	Set<GroundTerm> bboxTerms = new HashSet<GroundTerm>();
	Set<GroundTerm> actionTerms = new HashSet<GroundTerm>();
	Map<Variable, Set<GroundTerm>> subs = new HashMap<Variable, Set<GroundTerm>>();
	subs.put(BBox, bboxTerms);
	subs.put(Action, actionTerms);
	for (GroundAtom a : atoms) {
		bboxTerms.add(a.getArguments()[0]);
	}
	for (int a : actions) {
		actionTerms.add(new IntegerAttribute(a));
	}
	dbPop = new DatabasePopulator(trainDB);
	dbPop.populate(new QueryAtom(doing, BBox, Action), subs);
	
	/* Populate sameObj predicate.*/
	ArrayList<ArrayList<GroundTerm>>[] boxTerms = new ArrayList<ArrayList<GroundTerm>>[folds];
	atoms = Queries.getAllAtoms(trainDB, inFrame);
	for (GroundAtom a : atoms) {
		GroundTerm[] terms = a.getArguments();
		int s = ((IntegerAttribute)terms[1]).getValue().intValue() - 1;
		int f = ((IntegerAttribute)terms[2]).getValue().intValue() - 1;
		if (boxTerms[s] == null)
			boxTerms[s] = new ArrayList<ArrayList<GroundTerm>>();
		if (boxTerms[s].get(f) == null)
			boxTerms[s].set(f) = new ArrayList<GroundTerm>();
		boxTerms[s].get(f).add(a);
	}
	
	
	/* Labels DB */
	Partition label_wr = data.getNextPartition();
	Database labelDB = data.getDatabase(label_wr, [doing,sameObj] as Set, (Partition[])trainPartsLab.toArray());
	
	return 0;

	/*** EXPERIMENT ***/
	
	log.info("Starting experiment ...");
	
	for (int configIndex = 0; configIndex < configs.size(); configIndex++) {
		ConfigBundle config = configs.get(configIndex);
		def configName = config.getString("name", "");
		def method = config.getString("learningmethod", "");

		/* Weight learning */
		WeightLearner.learn(method, m, trainDB, labelsDB, initWeights, config, log)

		log.info("Learned model {}: \n {}", configName, m.toString())

		/* Inference on test set */
		Database predDB = data.getDatabase(write_te, toClose, read_te);
//		Set<GroundAtom> allAtoms = Queries.getAllAtoms(predDB, rating)
//		for (RandomVariableAtom atom : Iterables.filter(allAtoms, RandomVariableAtom))
//			atom.setValue(0.0)
		MPEInference mpe = new MPEInference(m, predDB, config)
		FullInferenceResult result = mpe.mpeInference()
		log.info("Objective: {}", result.getTotalWeightedIncompatibility())
		predDB.close();
	
		/* Evaluation */
		predDB = data.getDatabase(write_te);
//		Database groundTruthDB = data.getDatabase(labels_te, [rating] as Set)
//		def comparator = new ContinuousPredictionComparator(predDB)
//		comparator.setBaseline(groundTruthDB)
//		def metrics = [ContinuousPredictionComparator.Metric.MSE, ContinuousPredictionComparator.Metric.MAE]
//		double [] score = new double[metrics.size()]
//		for (int i = 0; i < metrics.size(); i++) {
//			comparator.setMetric(metrics.get(i))
//			score[i] = comparator.compare(rating)
//		}
		log.warn("Fold {} : {} : MSE {} : MAE {}", fold, configName, score[0], score[1]);
		expResults.get(config).add(fold, score);
		predDB.close();
		groundTruthDB.close()
	}
	trainDB.close()
}

log.warn("\n\nRESULTS\n");
for (ConfigBundle config : configs) {
	def configName = config.getString("name", "")
	def scores = expResults.get(config);
	for (int fold = 0; fold < folds; fold++) {
		def score = scores.get(fold)
		log.warn("{} \t{}\t{}\t{}", configName, fold, score[0], score[1]);
	}
}
