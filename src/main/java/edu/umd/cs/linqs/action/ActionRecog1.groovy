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
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.FullInferenceResult
import edu.umd.cs.psl.evaluation.statistics.ConfusionMatrix
import edu.umd.cs.psl.evaluation.statistics.MulticlassPredictionComparator
import edu.umd.cs.psl.evaluation.statistics.MulticlassPredictionStatistics
import edu.umd.cs.psl.groovy.*
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


//ConfusionMatrix cmat = new ConfusionMatrix(3);
//cmat.set(0,0,500);
//cmat.set(0,1,300);
//cmat.set(0,2,200);
//cmat.set(1,1,200);
//cmat.set(2,1,400);
//System.out.println(cmat.toMatlabString());
//SquareMatrix pmat = cmat.getPrecisionMatrix();
//System.out.println(pmat);
//SquareMatrix rmat = cmat.getRecallMatrix();
//System.out.println(rmat);
//System.out.println(SquareMatrix.average([pmat,rmat]).toMatlabString(5))
//return 0;

/*** CONFIGURATION PARAMETERS ***/

Logger log = LoggerFactory.getLogger(this.class)
ConfigManager cm = ConfigManager.getManager();
ConfigBundle cb = cm.getBundle("action");

def defPath = "data/action/action";// System.getProperty("java.io.tmpdir") + "/action"
def dbpath = cb.getString("dbpath", defPath)
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, false), cb)

int numSeqs = 44;

def sq = cb.getBoolean("squared", true);


/** EXPERIMENT CONFIG **/

ExperimentConfigGenerator configGenerator = new ExperimentConfigGenerator("action");

configGenerator.setModelTypes(["quad"]);

/* Learning methods */
//methods = ["MLE","MPLE","MM"];
methods = ["MLE"];
configGenerator.setLearningMethods(methods);

/* MLE/MPLE options */
configGenerator.setVotedPerceptronStepCounts([20]);
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

// FUNCIONAL PREDICATES
m.add function: "close", implementation: new ClosenessFunction(1e5, 1e-1);

/* FUNCTIONAL CONSTRAINTS */

// Functional constraint on doing means that it should sum to 1 for each BB
m.add PredicateConstraint.Functional, on: doing;

// Partial functional constraint on sameObj
m.add PredicateConstraint.PartialFunctional, on: sameObj;

/* ACTION RULES */

for (int a1 : actions) {

	// HOG-based SVM probabilities
//	m.add rule: hogAction(BB,a1) >> doing(BB,a1), weight: 1.0, squared: sq;
	
	// ACD-based SVM probabilities
	m.add rule: acdAction(BB,a1) >> doing(BB,a1), weight: 0.8, squared: sq;

	// Continuity of actions
	// If BB1,BB2 (in sequential frames) are the same object, and BB1 is doing action a1, then BB2 is doing action a1.
	m.add rule: ( sameObj(BB1,BB2) & doing(BB1,a1) ) >> doing(BB2,a1), weight: 1.0, squared: sq;

	// Action transitions
	for (int a2 : actions) {
		if (a1 == a2) continue;
		// If BB1,BB2 (in sequential frames) are the same object, and BB1 is doing action a1, then BB2 is doing action a2.
		m.add rule: ( sameObj(BB1,BB2) & doing(BB1,a1) ) >> doing(BB2,a2), weight: 0.7, squared: sq;
	}

	// Effect of proximity on actions
	for (int a2 : actions) {
		// If BB1,BB2 in same frame, and BB1 is doing action a1, and BB2 is close, then BB2 is doing action a2.
		m.add rule: ( inSameFrame(BB1,BB2) & doing(BB1,a1) & dims(BB1,X1,Y1,W1,H1) & dims(BB2,X2,Y2,W2,H2) & close(X1,X2,Y1,Y2,W1,W2,H1,H2) ) >> doing(BB2,a2), weight: 0.2, squared: sq;
	}
}

/* ID MAINTENANCE: BETWEEN-FRAME RULES */

// If BB1 in F1, BB2 in F2, and F1,F2 are sequential, and BB1,BB2 are close, then BB1,BB2 are same object.
m.add rule: ( inSeqFrames(BB1,BB2) & dims(BB1,X1,Y1,W1,H1) & dims(BB2,X2,Y2,W2,H2) & close(X1,X2,Y1,Y2,W1,W2,H1,H2) ) >> sameObj(BB1,BB2), weight: 0.25, squared: sq;

// Prior on sameObj
m.add rule: ~sameObj(BB1,BB2), weight: 0.2, squared: sq;

log.info("Model: {}", m)

/* get all default weights */
Map<CompatibilityKernel,Weight> initWeights = new HashMap<CompatibilityKernel, Weight>();
for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
	initWeights.put(k, k.getWeight());


/** DATASTORE PARTITIONS **/

int partCnt = 0;
Partition[][] partitions = new Partition[2][numSeqs];
for (int s = 0; s < numSeqs; s++) {
	partitions[0][s] = new Partition(partCnt++);	// observations
	partitions[1][s] = new Partition(partCnt++);	// labels
}
	
	
/** GLOBAL DATA FOR DB POPULATION **/

List<List<GroundTerm>> bboxesInSeq = new ArrayList<List<GroundTerm>>();
List<List<GroundTerm[]>> potentialMatchesInSeq = new ArrayList<List<GroundTerm[]>>();
for (int s = 0; s < numSeqs; s++) {
	bboxesInSeq.add(new ArrayList<GroundTerm>());
	potentialMatchesInSeq.add(new ArrayList<GroundTerm[]>());
}
def toClose = [inFrame, inSameFrame, dims, hogAction, acdAction, inSeqFrames] as Set;
Database db = data.getDatabase(new Partition(partCnt++), toClose, partitions[0]);
Set<GroundAtom> atoms = Queries.getAllAtoms(db, inFrame);
for (GroundAtom a : atoms) {
	GroundTerm[] terms = a.getArguments();
	int s = ((IntegerAttribute)terms[1]).getValue().intValue() - 1;
	bboxesInSeq[s].add(terms[0]);
}
atoms = Queries.getAllAtoms(db, inSeqFrames);
for (GroundAtom a : atoms) {
	GroundTerm[] terms = a.getArguments();
	int s = Integer.parseInt(terms[0].toString()) / 100000 - 1;
	potentialMatchesInSeq[s].add(terms);
}
db.close();
Map<GroundTerm,Integer> actionMap = new HashMap<GroundTerm,Integer>();
for (int i = 0; i < actions.size(); i++) {
	actionMap.put(new IntegerAttribute(actions[i]), i);
}


/*** RUN EXPERIMENTS ***/

log.info("Starting experiments.");

int numFolds = 44;

Map<String, List<MulticlassPredictionStatistics>> stats_doing = new HashMap<String, List<MulticlassPredictionStatistics>>()
for (ConfigBundle method : configs)
	stats_doing.put(method, new ArrayList<MulticlassPredictionStatistics>())

for (int fold = 0; fold < numFolds; fold++) {
//for (int fold = 0; fold < 2; fold++) {
	
	log.info("\n\n*** STARTING FOLD {} ***\n", fold)
		
	/** SPLIT DATA **/
	
	log.info("Splitting data ...");
	
	/* To construct training set: query for all of the atoms from each scene, except for hold-out. */
	List<Partition> trainPartsObs = new ArrayList<Partition>();
	List<Partition> trainPartsLab = new ArrayList<Partition>();
	List<Partition> testPartsObs = new ArrayList<Partition>();
	List<Partition> testPartsLab = new ArrayList<Partition>();
	for (int s = 0; s < numSeqs; s++) {
		if ((s+fold) % numFolds != 0) {
			trainPartsObs.add(partitions[0][s]);
			trainPartsLab.add(partitions[1][s]);
		}
		else {
			testPartsObs.add(partitions[0][s]);
			testPartsLab.add(partitions[1][s]);
		}
	}
	
	/** POPULATE DB ***/

	log.info("Populating databases ...");
		
	Partition write_tr = new Partition(partCnt++);
	Partition write_te = new Partition(partCnt++);
	Database trainDB = data.getDatabase(write_tr, toClose, (Partition[])trainPartsObs.toArray());
	Database testDB = data.getDatabase(write_te, toClose, (Partition[])testPartsObs.toArray());

	/* Populate doing predicate. */
	Variable BBox = new Variable("BBox");
	Variable Action = new Variable("Action");
	Map<Variable, Set<GroundTerm>> subs = new HashMap<Variable, Set<GroundTerm>>();
	subs.put(Action, actionMap.keySet());
	// Get all bbox ground terms
	Set<GroundTerm> bboxTerms_tr = new HashSet<GroundTerm>();
	Set<GroundTerm> bboxTerms_te = new HashSet<GroundTerm>();
	for (int s = 0; s < numSeqs; s++) {
		Set<GroundTerm> curSet = ((s+fold) % numFolds != 0) ? bboxTerms_tr : bboxTerms_te;
		curSet.addAll(bboxesInSeq[s]);
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
	for (int s = 0; s < numSeqs; s++) {
		if ((s+fold) % numFolds != 0) {
			for (GroundTerm[] terms : potentialMatchesInSeq[s]) {
				RandomVariableAtom rv = (RandomVariableAtom)trainDB.getAtom(sameObj, terms[0], terms[1]);
				trainDB.commit(rv);
			}
		}
		else {
			for (GroundTerm[] terms : potentialMatchesInSeq[s]) {
				RandomVariableAtom rv = (RandomVariableAtom)testDB.getAtom(sameObj, terms[0], terms[1]);
				testDB.commit(rv);
				++numTestEx_sameObj;
			}
		}
	}	

	/* Need to close testDB so that we can use write_te for multiple databases. */	
	testDB.close();
	
	/* Label DBs */
	Database labelDB = data.getDatabase(new Partition(partCnt++), [doing,sameObj] as Set, (Partition[])trainPartsLab.toArray());
	Database truthDB = data.getDatabase(new Partition(partCnt++), [doing,sameObj] as Set, (Partition[])testPartsLab.toArray());

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
		testDB = data.getDatabase(write_te, toClose, (Partition[])testPartsObs.toArray());
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
		Database predDB = data.getDatabase(write_te, [doing,sameObj] as Set);
		def comparator = new MulticlassPredictionComparator(predDB);
		comparator.setBaseline(truthDB);
		MulticlassPredictionStatistics stats = comparator.compare(doing, actionMap, 1);
		log.info("F1  ACTION: {}", stats.getF1());
		log.info("Acc ACTION: {}", stats.getAccuracy());
		ConfusionMatrix conMat = stats.getConfusionMatrix();
		System.out.println("Confusion Matrix:\n" + conMat.toMatlabString());
		System.out.println("Precision matrix:\n" + conMat.getPrecisionMatrix().toMatlabString(3));
		stats_doing.get(config).add(fold, stats);
		predDB.close();
	}
	
	/* Close all databases. */
	trainDB.close();
	labelDB.close();
	truthDB.close();
	
	/* Empty the write partitions */
	data.deletePartition(write_tr);
	data.deletePartition(write_te);
}

log.info("\n\nRESULTS\n");
for (ConfigBundle config : configs) {
	String configName = config.getString("name", "")
	def stats = stats_doing.get(config);
	double avgF1 = 0.0;
	double avgAcc = 0.0;
	List<ConfusionMatrix> cmats = new ArrayList<ConfusionMatrix>();
//	List<SquareMatrix> pmats = new ArrayList<SquareMatrix>();
	for (int fold = 0; fold < stats.size(); fold++) {
		avgF1 += stats.get(fold).getF1();
		avgAcc += stats.get(fold).getAccuracy();
		ConfusionMatrix cmat = stats.get(fold).getConfusionMatrix();
		cmats.add(cmat)
//		pmats.add(cmat.getPrecisionMatrix());
	}
	/* Average statistics */
	avgF1 /= stats.size();
	avgAcc /= stats.size();
	log.info("\n{}\n Avg F1: {}\n Avg Acc: {}\n", configName, avgF1, avgAcc);
//	SquareMatrix avgPMat = SquareMatrix.average(pmats);
//	log.info("\nAvg Precision Matrix:\n{}", avgPMat.toMatlabString(3));
	/* Cummulative statistics */
	ConfusionMatrix cumCMat = ConfusionMatrix.aggregate(cmats);
	def cumStats = new MulticlassPredictionStatistics(cumCMat);
	log.info("\n{}\n Cumm F1: {}\n Cumm Acc: {}\n", configName, cumStats.getF1(), cumStats.getF1());
	log.info("\nCumm Precision Matrix:\n{}", cumCMat.getPrecisionMatrix().toMatlabString(3));
}



