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
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.FullInferenceResult
import edu.umd.cs.psl.evaluation.statistics.ContinuousPredictionComparator
import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.GroundTerm
import edu.umd.cs.psl.model.argument.UniqueID
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
methods = ["MLE","MPLE","MM"];
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
m.add predicate: "frame", types: [ArgumentType.UniqueID];
m.add predicate: "bbox", types: [ArgumentType.UniqueID];

// target
m.add predicate: "doing", types: [ArgumentType.UniqueID,ArgumentType.Integer];
m.add predicate: "sameObj", types: [ArgumentType.UniqueID,ArgumentType.UniqueID];

// observed
m.add predicate: "adjFrames", types: [ArgumentType.UniqueID,ArgumentType.UniqueID];
m.add predicate: "inFrame", types: [ArgumentType.UniqueID,ArgumentType.UniqueID];
m.add predicate: "inSameFrame", types: [ArgumentType.UniqueID,ArgumentType.UniqueID];
m.add predicate: "dims", types: [ArgumentType.UniqueID,ArgumentType.Integer,ArgumentType.Integer,ArgumentType.Integer,ArgumentType.Integer];
m.add predicate: "hogAction", types: [ArgumentType.UniqueID,ArgumentType.Integer];
m.add predicate: "acdAction", types: [ArgumentType.UniqueID,ArgumentType.Integer];

// derived
m.add function: "near", implementation: new NearFunction();
m.add function: "far", implementation: new DistanceFunction();

/* RULES */

// In-frame rules
for (int a1 : actions) {
	// HOG-based SVM probabilities
	m.add rule: hogAction(BB,a1) >> doing(BB,a1), weight: 1.0, squared: sq;
	// ACD-based SVM probabilities
	m.add rule: acdAction(BB,a1) >> doing(BB,a1), weight: 1.0, squared: sq;
	// Relational rules
	for (int a2 : actions) {
		// If BB1,BB2 in same frame, and BB1 is doing action a1, and BB2 is NEAR, then BB2 is doing action a2.
		m.add rule: ( inSameFrame(BB1,BB2) & doing(BB1,a1) & dims(BB1,X1,Y1,W1,H1) & dims(BB2,X2,Y2,W2,H2) & near(X1,X2,Y1,Y2,W1,W2,H1,H2) ) >> doing(BB2,a2), weight: 1.0, squared: sq;
		// If BB1,BB2 in same frame, and BB1 is doing action a1, and BB2 is FAR, then BB2 is doing action a2.
		m.add rule: ( inSameFrame(BB1,BB2) & doing(BB1,a1) & dims(BB1,X1,Y1,W1,H1) & dims(BB2,X2,Y2,W2,H2) & far(X1,X2,Y1,Y2,W1,W2,H1,H2) ) >> doing(BB2,a2), weight: 1.0, squared: sq;
	}
	//TODO: Priors on actions?
}

// Between-frame rules
m.add rule: (  ) >> sameObj(BB1,BB2), weight: 1.0, squared: sq;

log.info("Model: {}", m)

/* get all default weights */
Map<CompatibilityKernel,Weight> initWeights = new HashMap<CompatibilityKernel, Weight>();
for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
	initWeights.put(k, k.getWeight());

return 0;

/*** LOAD DATA ***/

log.info("Loading data ...");

Map<ConfigBundle,ArrayList<Double>> expResults = new HashMap<String,ArrayList<Double>>();
for (ConfigBundle config : configs) {
	expResults.put(config, new ArrayList<Double>(folds));
}

for (int fold = 0; fold < folds; fold++) {

	Partition read_tr = new Partition(0 + fold * folds);
	Partition write_tr = new Partition(1 + fold * folds);
	Partition read_te = new Partition(2 + fold * folds);
	Partition write_te = new Partition(3 + fold * folds);
	Partition labels_tr = new Partition(4 + fold * folds);
	Partition labels_te = new Partition(5 + fold * folds);

	/** LOAD FILES **/
	
	def inserter;	

	/** POPULATE DB ***/


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
