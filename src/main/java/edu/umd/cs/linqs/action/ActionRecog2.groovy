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
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionComparator
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



/*** CONFIGURATION PARAMETERS ***/

Logger log = LoggerFactory.getLogger(this.class)
ConfigManager cm = ConfigManager.getManager();
ConfigBundle cb = cm.getBundle("action");

//def defPath = "data/action/action";
def defPath = System.getProperty("java.io.tmpdir") + "/action2"
def dbpath = cb.getString("dbpath", defPath)
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, false), cb)

def outPath = "output/action2_loo_hog/";
int numFolds = 63;

int numSeqs = 63;

def sq = cb.getBoolean("squared", true);
def computeBaseline = true;
def baselineMethod = cb.getString("baseline", "acd");
def discreteModel = cb.getBoolean("discrete", false);

/* Which fold are we running? */
int startFold = 0;
int endFold = numFolds;
if (args.length >= 1) {
	startFold = Integer.parseInt(args[0]);
	endFold = startFold + 1;
}


/*** EXPERIMENT CONFIG ***/

ExperimentConfigGenerator configGenerator = new ExperimentConfigGenerator("action");

if (discreteModel) {
	configGenerator.setModelTypes(["bool"]);
	sq = false;
}
else
	configGenerator.setModelTypes(["quad"]);

/* Learning methods */
methods = ["MLE"];
configGenerator.setLearningMethods(methods);

/* MLE/MPLE options */
configGenerator.setVotedPerceptronStepCounts([5,50]);
configGenerator.setVotedPerceptronStepSizes([(double) 0.1, (double) 1.0]);

/* MM options */
//configGenerator.setMaxMarginSlackPenalties([(double) 0.1]);
//configGenerator.setMaxMarginLossBalancingTypes([LossBalancingType.NONE]);
//configGenerator.setMaxMarginNormScalingTypes([NormScalingType.NONE]);
//configGenerator.setMaxMarginSquaredSlackValues([false]);

List<ConfigBundle> configs = configGenerator.getConfigs();
for (ConfigBundle config : configs)
	System.out.println(config);


/*** MODEL DEFINITION ***/

log.info("Initializing model ...");

PSLModel m = new PSLModel(this, data);

/* PREDICATES (STORED IN DB) */

def obsPreds = [inFrame, inSameFrame, inSeqFrames, dims, hogAction, acdAction, hogFrameAction, acdFrameAction] as Set;
def targetPreds = [doing, sameObj] as Set;

/* CONSTANTS */

def actions = [1,2,3,5,6,7];
def actionNames = ["crossing","standing","queueing","talking","dancing","jogging"];
int numHogs = 75;

// FUNCIONAL PREDICATES
m.add function: "close", implementation: new ClosenessFunction(0, 1e6, 0.1, true);
//m.add function: "seqClose", implementation: new ClosenessFunction(0, 400, 0.7, true);
m.add function: "seqClose", implementation: new ClosenessFunction(100, 4.0, 0.7, true);
m.add function: "notMoved", implementation: new ClosenessFunction(10, 1.0, 0.0, false);

/* FUNCTIONAL CONSTRAINTS */

// Functional constraint on doing means that it should sum to 1 for each BB
m.add PredicateConstraint.Functional, on: doing;

// (Inverse) Partial functional constraint on sameObj
m.add PredicateConstraint.PartialFunctional, on: sameObj;
if (!discreteModel)
	m.add PredicateConstraint.PartialInverseFunctional, on: sameObj;

/* ID MAINTENANCE: BETWEEN-FRAME RULES */

// If BB1 in F1, BB2 in F2, and F1,F2 are sequential, and BB1,BB2 are close, then BB1,BB2 are same object.
m.add rule: ( inSeqFrames(BB1,BB2) & dims(BB1,X1,Y1,W1,H1) & dims(BB2,X2,Y2,W2,H2) & seqClose(X1,X2,Y1,Y2,W1,W2,H1,H2) ) >> sameObj(BB1,BB2), weight: 1.0, squared: sq;

// Prior on sameObj
m.add rule: ~sameObj(BB1,BB2), weight: 0.01, squared: sq;

/* ACTION RULES */

for (int a1 : actions) {

	if (baselineMethod.equals("hog")) {
		// HOG-based SVM probabilities
		m.add rule: hogAction(BB,a1) >> doing(BB,a1), weight: 1.0, squared: sq;
		// Frame label
		m.add rule: ( inFrame(BB,S,F) & hogFrameAction(F,a1) ) >> doing(BB,a1), weight: 0.1, squared: sq;
	}
	else {
		// ACD-based SVM probabilities
		m.add rule: acdAction(BB,a1) >> doing(BB,a1), weight: 1.0, squared: sq;
		// Frame label
		m.add rule: ( inFrame(BB,S,F) & acdFrameAction(F,a1) ) >> doing(BB,a1), weight: 0.1, squared: sq;
	}

	// Continuity of actions
	// If BB1,BB2 (in sequential frames) are the same object, and BB1 is doing action a1, then BB2 is doing action a1.
	m.add rule: ( sameObj(BB1,BB2) & doing(BB1,a1) ) >> doing(BB2,a1), weight: 1.0, squared: sq;
	
	// Effect of proximity on actions
	m.add rule: ( inSameFrame(BB1,BB2) & doing(BB1,a1) & dims(BB1,X1,Y1,W1,H1) & dims(BB2,X2,Y2,W2,H2) & close(X1,X2,Y1,Y2,W1,W2,H1,H2) ) >> doing(BB2,a1), weight: 0.1, squared: sq;
	
	// Stationary vs. mobile actions
//	if (a1 in [2,3,4])
//		m.add rule: ( sameObj(BB1,BB2) & dims(BB1,X1,Y1,W1,H1) & dims(BB2,X2,Y2,W2,H2) & notMoved(X1,X2,Y1,Y2,W1,W2,H1,H2) ) >> doing(BB1,a1), weight: 0.1, squared: sq;
//	else
//		m.add rule: ( sameObj(BB1,BB2) & dims(BB1,X1,Y1,W1,H1) & dims(BB2,X2,Y2,W2,H2) & ~notMoved(X1,X2,Y1,Y2,W1,W2,H1,H2) ) >> doing(BB1,a1), weight: 0.1, squared: sq;

	// Action transition
//	m.add rule: ( sameObj(BB1,BB2) & doing(BB1,a1) ) >> ~doing(BB2,a1), weight: 0.2, squared: sq;
//	for (int a2 : actions) {
//		if (a1 == a2) continue;
//		// If BB1,BB2 (in sequential frames) are the same object, and BB1 is doing action a1, then BB2 is doing action a2.
//		m.add rule: ( sameObj(BB1,BB2) & doing(BB1,a1) ) >> doing(BB2,a2), weight: 0.7, squared: sq;
//	}
	
	// Frame consistency of action
//	for (int a2 : actions) {
//		// If BB1,BB2 in same frame, and BB1 is doing action a1, and BB2 is close, then BB2 is doing action a2.
//		m.add rule: ( inSameFrame(BB1,BB2) & doing(BB1,a1) ) >> doing(BB2,a2), weight: 0.2, squared: sq;
//	}
}

//log.info("Model: {}", m)

/* get all default weights */
Map<CompatibilityKernel,Weight> initWeights = new HashMap<CompatibilityKernel, Weight>();
for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
	initWeights.put(k, k.getWeight());


/*** DATASTORE PARTITIONS ***/

int partCnt = 0;
Partition[][] partitions = new Partition[2][numSeqs];
for (int s = 0; s < numSeqs; s++) {
	partitions[0][s] = new Partition(partCnt++);	// observations
	partitions[1][s] = new Partition(partCnt++);	// labels
}

	
/*** GLOBAL DATA FOR DB POPULATION ***/

List<List<GroundTerm>> bboxesInSeq = new ArrayList<List<GroundTerm>>();
List<List<GroundTerm[]>> potentialMatchesInSeq = new ArrayList<List<GroundTerm[]>>();
for (int s = 0; s < numSeqs; s++) {
	bboxesInSeq.add(new ArrayList<GroundTerm>());
	potentialMatchesInSeq.add(new ArrayList<GroundTerm[]>());
}
Database db = data.getDatabase(new Partition(partCnt++), obsPreds, partitions[0]);
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

Map<String, List<MulticlassPredictionStatistics>> stats_doing = new HashMap<String, List<MulticlassPredictionStatistics>>()
for (ConfigBundle method : configs)
	stats_doing.put(method, new ArrayList<MulticlassPredictionStatistics>())
List<MulticlassPredictionStatistics> stats_base = new ArrayList<MulticlassPredictionStatistics>()
	
for (int fold = startFold; fold < endFold; fold++) {
	
	log.info("\n\n*** STARTING FOLD {} ***\n", fold)
		
	/*** SPLIT DATA ***/
	
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
	
	/*** POPULATE DB ***/

	log.info("Populating databases ...");
		
	Partition write_tr = new Partition(partCnt++);
	Partition write_te = new Partition(partCnt++);
	Database trainDB = data.getDatabase(write_tr, obsPreds, (Partition[])trainPartsObs.toArray());
	Database testDB = data.getDatabase(write_te, obsPreds, (Partition[])testPartsObs.toArray());

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
				GroundAtom rv = trainDB.getAtom(sameObj, terms[0], terms[1]);
				if (rv instanceof RandomVariableAtom)
					trainDB.commit((RandomVariableAtom)rv);
			}
		}
		else {
			for (GroundTerm[] terms : potentialMatchesInSeq[s]) {
				GroundAtom rv = testDB.getAtom(sameObj, terms[0], terms[1]);
				if (rv instanceof RandomVariableAtom)
					testDB.commit((RandomVariableAtom)rv);
				++numTestEx_sameObj;
			}
		}
	}

	/* Need to close testDB so that we can use write_te for multiple databases. */	
	testDB.close();
	
	/* Label DBs */
	Database labelDB = data.getDatabase(new Partition(partCnt++), targetPreds, (Partition[])trainPartsLab.toArray());
	Database truthDB = data.getDatabase(new Partition(partCnt++), targetPreds, (Partition[])testPartsLab.toArray());
	
	/* Compute baseline accuracy using ACDs. */
	if (computeBaseline) {
		Partition write_base = new Partition(partCnt++);
		Database baselineDB = data.getDatabase(write_base, obsPreds, (Partition[])testPartsObs.toArray());
		if (baselineMethod.equals("hog"))
			atoms = Queries.getAllAtoms(baselineDB, hogAction);
		else
			atoms = Queries.getAllAtoms(baselineDB, acdAction);
		for (GroundAtom a : atoms) {
			GroundTerm[] terms = a.getArguments();
			RandomVariableAtom rv = (RandomVariableAtom)baselineDB.getAtom(doing, terms);
			rv.setValue(a.getValue());
			baselineDB.commit(rv);
		}
		def compBaseline = new MulticlassPredictionComparator(baselineDB);
		compBaseline.setBaseline(truthDB);
		def baselineStats = compBaseline.compare(doing, actionMap, 1);
		log.info("ACTION ACC: {}", baselineStats.getAccuracy());
		log.info("ACTION F1:  {}", baselineStats.getF1());
		ConfusionMatrix baselineConMat = baselineStats.getConfusionMatrix();
		System.out.println("Confusion Matrix:\n" + baselineConMat.toMatlabString());
		stats_base.add(baselineStats);
		baselineDB.close();
		data.deletePartition(write_base);
	}

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
		testDB = data.getDatabase(write_te, obsPreds, (Partition[])testPartsObs.toArray());
		Set<GroundAtom> targetAtoms = Queries.getAllAtoms(testDB, doing)
		for (RandomVariableAtom rv : Iterables.filter(targetAtoms, RandomVariableAtom))
			rv.setValue(0.0)
		targetAtoms = Queries.getAllAtoms(testDB, sameObj);
		for (RandomVariableAtom rv : Iterables.filter(targetAtoms, RandomVariableAtom))
			rv.setValue(0.0)
		MPEInference mpe = new MPEInference(m, testDB, config)
		FullInferenceResult result = mpe.mpeInference()
		log.info("Objective: {}", result.getTotalWeightedIncompatibility())
		mpe.close();
		testDB.close();

		/* Open the prediction DB. */
		Database predDB = data.getDatabase(write_te, targetPreds);
		
		/* Evaluate doing predicate */
		def comparator = new MulticlassPredictionComparator(predDB);
		comparator.setBaseline(truthDB);
		def stats = comparator.compare(doing, actionMap, 1);
		log.info("ACTION ACC: {}", stats.getAccuracy());
		log.info("ACTION F1:  {}", stats.getF1());
		ConfusionMatrix conMat = stats.getConfusionMatrix();
		System.out.println("Confusion Matrix:\n" + conMat.toMatlabString());
		stats_doing.get(config).add(stats);
		
		/* Write confusion matrix to file. */
		File outFile = new File(outPath);
		outFile.mkdirs();
		FileOutputStream fileStr = new FileOutputStream(outPath + configName.replace('.', '_') + "-fold" + fold + ".matrix");
		ObjectOutputStream objOutStr = new ObjectOutputStream(fileStr);
		objOutStr.writeObject(conMat);
		
		/* Evaluate sameObj predicate. */
		if (sameObj in targetPreds) {
			int numThresh = 4;
			comparator = new DiscretePredictionComparator(predDB);
			comparator.setBaseline(truthDB);
			for (int th = 0; th < numThresh; th++) {
				double thresh = (th+1.0) / (numThresh+1.0)
				comparator.setThreshold(thresh);
				def sameObjStats = comparator.compare(sameObj);
				log.info("SAME_OBJ Th: {} ACC: {}", thresh, sameObjStats.getAccuracy());
				log.info("SAME_OBJ Th: {} F1:  {}", thresh, sameObjStats.getF1());
				log.info("SAME_OBJ Th: {} PRE: {}", thresh, sameObjStats.getPrecision());
				log.info("SAME_OBJ Th: {} REC: {}", thresh, sameObjStats.getRecall());
			}
		}		
		/* Close the prediction DB. */
		predDB.close();
	}
	
	/* Close all databases. */
	trainDB.close();
	labelDB.close();
	truthDB.close();
	
	/* Empty the write partitions */
	data.deletePartition(write_tr);
	data.deletePartition(write_te);
	
	/* Give the GC a stern suggestion that it should take out the trash. */
	//System.gc();
}


/*** PRINT RESULTS ***/

/* Only run this block if we're doing all folds at once. */
if (args.size() == 0 && computeBaseline) {
	log.info("\n\nBASELINE RESULTS\n");
	double avgF1 = 0.0;
	double avgAcc = 0.0;
	List<ConfusionMatrix> cmats = new ArrayList<ConfusionMatrix>();
	for (int i = 0; i < stats_base.size(); i++) {
		avgF1 += stats_base.get(i).getF1();
		avgAcc += stats_base.get(i).getAccuracy();
		ConfusionMatrix cmat = stats_base.get(i).getConfusionMatrix();
		cmats.add(cmat)
	}
	/* Average statistics */
	avgF1 /= stats_base.size();
	avgAcc /= stats_base.size();
	log.info("\nBaseline\n Avg Acc: {}\n Avg F1:  {}\n", avgAcc, avgF1);
	/* Cummulative statistics */
	ConfusionMatrix cumCMat = ConfusionMatrix.aggregate(cmats);
	def cumStats = new MulticlassPredictionStatistics(cumCMat);
	log.info("\nBaseline\n Cum Acc: {}\n Cum F1:  {}\n", cumStats.getAccuracy(), cumStats.getF1());
	log.info("\nCum Recall Matrix:\n{}", cumCMat.getRecallMatrix().toMatlabString(3));
}

/* Only run this block if we're doing all folds at once. */
if (args.size() == 0) {
	log.info("\n\nRESULTS\n");
	for (ConfigBundle config : configs) {
		String configName = config.getString("name", "")
		def stats = stats_doing.get(config);
		double avgF1 = 0.0;
		double avgAcc = 0.0;
		List<ConfusionMatrix> cmats = new ArrayList<ConfusionMatrix>();
		for (int i = 0; i < stats.size(); i++) {
			avgF1 += stats.get(i).getF1();
			avgAcc += stats.get(i).getAccuracy();
			ConfusionMatrix cmat = stats.get(i).getConfusionMatrix();
			cmats.add(cmat)
		}
		/* Average statistics */
		avgF1 /= stats.size();
		avgAcc /= stats.size();
		log.info("\n{}\n Avg Acc: {}\n Avg F1:  {}\n", configName, avgAcc, avgF1);
		/* Cummulative statistics */
		ConfusionMatrix cumCMat = ConfusionMatrix.aggregate(cmats);
		def cumStats = new MulticlassPredictionStatistics(cumCMat);
		log.info("\n{}\n Cum Acc: {}\n Cum F1:  {}\n", configName, cumStats.getAccuracy(), cumStats.getF1());
		log.info("\nCum Recall Matrix:\n{}", cumCMat.getRecallMatrix().toMatlabString(3));
	}
}

