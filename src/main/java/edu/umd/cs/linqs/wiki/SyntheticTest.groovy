package edu.umd.cs.linqs.wiki

import org.slf4j.Logger
import org.slf4j.LoggerFactory
import com.google.common.collect.Iterables

import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.LossBalancingType
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.NormScalingType
import edu.umd.cs.psl.application.learning.weight.random.FirstOrderMetropolisRandOM
import edu.umd.cs.psl.application.learning.weight.random.GroundMetropolisRandOM
import edu.umd.cs.psl.application.learning.weight.random.MetropolisRandOM
import edu.umd.cs.psl.application.learning.weight.random.IncompatibilityMetropolisRandOM
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.VotedPerceptron
import edu.umd.cs.psl.config.ConfigBundle;
import edu.umd.cs.psl.config.ConfigManager
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.loading.Inserter
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.FullInferenceResult
import edu.umd.cs.psl.groovy.PSLModel
import edu.umd.cs.psl.model.Model;
import edu.umd.cs.psl.model.argument.UniqueID
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.atom.GroundAtom
import edu.umd.cs.psl.model.atom.RandomVariableAtom
import edu.umd.cs.psl.model.kernel.CompatibilityKernel
import edu.umd.cs.psl.model.parameters.Weight


/*
 * SET LEARNING ALGORITHMS
 * 
 * Options:
 * "MLE" (MaxLikelihoodMPE)
 * "MPLE" (MaxPseudoLikelihood)
 * "MM" (MaxMargin)
 * "FirstOrderRandOM" (FirstOrderMetropolisRandOM)
 * "IncompatibilityRandOM" (IncompatibilityMetropolisRandOM)
 * "GroundRandOM" (GroundMetropolisRandOM)
 */
methods = ["MM"];

/* MLE/MPLE options */
vpStepCounts = [2000]
vpStepSizes = [5]

/* MM options */
slackPenalties = [1]
lossBalancings = [LossBalancingType.NONE]
normScalings = [NormScalingType.NONE]
//slackPenalties = [1, 10, 100]
//lossBalancings = [LossBalancingType.NONE, LossBalancingType.CLASS_WEIGHTS, LossBalancingType.INVERSE_CLASS_WEIGHTS]
//normScalings = [NormScalingType.NONE, NormScalingType.NUM_GROUNDINGS, NormScalingType.INVERSE_NUM_GROUNDINGS]

/* Metropolis RandOM options */
sampleCounts = [50]
burnInFractions = [0.1]
maxIters = [25]
obsvScales = [1]

Logger log = LoggerFactory.getLogger(this.class)

ConfigManager cm = ConfigManager.getManager()
ConfigBundle baseConfig = cm.getBundle("synth")

boolean sq = true

/*
 * DEFINES EXPERIMENT CONFIGURATIONS
 */
List<String> methodNames = new ArrayList<String>();
List<ConfigBundle> methodConfigs = new ArrayList<ConfigBundle>();
for (String method : methods) {
	if (method.equals("MLE") || method.equals("MPLE")) {
		for (int vpStepCount : vpStepCounts) {
			for (double vpStepSize : vpStepSizes) {
				ConfigBundle newBundle = cm.getBundle("synth");
				newBundle.addProperty("method", method);
				newBundle.addProperty(VotedPerceptron.NUM_STEPS_KEY, vpStepCount);
				newBundle.addProperty(VotedPerceptron.STEP_SIZE_KEY, vpStepSize);
				methodName = ((sq) ? "quad" : "linear") + "-" + method.toLowerCase() + "-" + vpStepCount + "-" + vpStepSize;
				methodNames.add(methodName);
				methodConfigs.add(newBundle);
			}
		}
	}
	else if (method.equals("MM")) {
		for (double slackPenalty : slackPenalties) {
			for (LossBalancingType lossBalancing : lossBalancings) {
				for (NormScalingType normScaling : normScalings) {
					ConfigBundle newBundle = cm.getBundle("synth");
					newBundle.addProperty("method", method);
					newBundle.addProperty(MaxMargin.SLACK_PENALTY_KEY, slackPenalty);
					newBundle.addProperty(MaxMargin.BALANCE_LOSS_KEY, lossBalancing);
					newBundle.addProperty(MaxMargin.SCALE_NORM_KEY, normScaling);
					methodName = ((sq) ? "quad" : "linear") + "-mm-" + slackPenalty + "-" + lossBalancing.name().toLowerCase() + "-" + normScaling.name().toLowerCase();
					methodNames.add(methodName);
					methodConfigs.add(newBundle);
				}
			}
		}
	}
	else if (method.equals("FirstOrderRandOM") || method.equals("IncompatibilityRandOM") || method.equals("GroundRandOM")) {
		for (int numSamples : sampleCounts) {
			for (double burnInFraction : burnInFractions) {
				burnIn = Math.round(numSamples * burnInFraction);
				for (int maxIter : maxIters) {
					for (double obsvScale : obsvScales) {
						ConfigBundle newBundle = cm.getBundle("synth");
						newBundle.addProperty("method", method);
						newBundle.addProperty(GroundMetropolisRandOM.PROPOSAL_VARIANCE, 0.00005);
						newBundle.addProperty(MetropolisRandOM.INITIAL_VARIANCE_KEY, 1);
						newBundle.addProperty(MetropolisRandOM.CHANGE_THRESHOLD_KEY, 0.001);
						newBundle.addProperty(MetropolisRandOM.NUM_SAMPLES_KEY, numSamples);
						newBundle.addProperty(MetropolisRandOM.BURN_IN_KEY, burnIn);
						newBundle.addProperty(MetropolisRandOM.MAX_ITER_KEY, maxIter);
						newBundle.addProperty(MetropolisRandOM.OBSERVATION_DENSITY_SCALE_KEY, obsvScale);
						methodName = ((sq) ? "quad" : "linear") + "-" + method.toLowerCase() + "-" + numSamples + "-" + burnIn + "-" + maxIter + "-" + obsvScale;
						methodNames.add(methodName);
						methodConfigs.add(newBundle);
					}
				}
			}
		}
	}
	else {
		ConfigBundle newBundle = cm.getBundle("synth");
		newBundle.addProperty("method", method);
		methodName = ((sq) ? "quad" : "linear") + "-" + method.toLowerCase();
		methodNames.add(methodName);
		methodConfigs.add(newBundle);
	}
}

/*
 * PRINTS EXPERIMENT CONFIGURATIONS
 */
for (int methodIndex = 0; methodIndex < methodNames.size(); methodIndex++)
	System.out.println("Config for " + methodNames.get(methodIndex) + "\n" + methodConfigs.get(methodIndex));

/*
 * INITIALIZES DATASTORE AND MODEL
 */
def defaultPath = System.getProperty("java.io.tmpdir") + "/"
//def defaultPath = "/scratch0/bach-icml13/"
String dbpath = baseConfig.getString("dbpath", defaultPath + "pslSynthetic")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), baseConfig)

PSLModel m = new PSLModel(this, data)

/*
 * DEFINE MODEL
 */
m.add predicate: "label", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "dummy", types: [ArgumentType.UniqueID]

double initialWeight = 10

for (int i = 0; i < 10; i++) {
	UniqueID index = data.getUniqueID(i)
	m.add rule: (dummy(L)) >> label(index, L), weight: initialWeight, squared: sq
	m.add rule: (dummy(L)) >> ~label(index, L), weight: initialWeight, squared: sq
}


// save all initial weights
Map<CompatibilityKernel,Weight> weights = new HashMap<CompatibilityKernel, Weight>()
for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
	weights.put(k, k.getWeight());

Partition trainWrite = new Partition(0)
Partition trainLabel = new Partition(1)
Partition trainRead = new Partition(2)

/*
 * LOAD DATA
 */

Inserter insert = data.getInserter(dummy, trainRead)
insert.insert(data.getUniqueID(0))

Database trainDB = data.getDatabase(trainWrite, [dummy] as Set, trainRead)
Database labelsDB = data.getDatabase(trainLabel)

for (int i = 0; i < 10; i++) {
	UniqueID index = data.getUniqueID(i)
	UniqueID L = data.getUniqueID(0)
	RandomVariableAtom atom = (RandomVariableAtom) trainDB.getAtom(label, index, L)
	atom.setValue((i%3 == 0)? 1.0 : 0.0)
	labelsDB.commit(atom)
}
labelsDB.close()
labelsDB = data.getDatabase(trainLabel, [label] as Set)

// populate trainDB
for (int i = 0; i < 10; i++) {
	UniqueID index = data.getUniqueID(i)
	UniqueID L = data.getUniqueID(0)
	RandomVariableAtom atom = (RandomVariableAtom) trainDB.getAtom(label, index, L)
	trainDB.commit(atom)
}


for (int methodIndex = 0; methodIndex < methodNames.size(); methodIndex++) {
	for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
		k.setWeight(weights.get(k))

	/*
	 * Weight learning
	 */
	learn(m, trainDB, labelsDB, methodConfigs.get(methodIndex), log)

	System.out.println("Learned model " + methodNames.get(methodIndex) + "\n" + m.toString())

	// populate trainDB
	for (int i = 0; i < 10; i++) {
		UniqueID index = data.getUniqueID(i)
		UniqueID L = data.getUniqueID(0)
		RandomVariableAtom atom = (RandomVariableAtom) trainDB.getAtom(label, index, L)
		atom.setValue(0.0)
		trainDB.commit(atom)
	}
	
	MPEInference mpe = new MPEInference(m, trainDB, baseConfig)
	FullInferenceResult result = mpe.mpeInference()

	/*
	 * Evaluation
	 * implement area under PR curve
	 */

	for (int i = 0; i < 10; i++) {
		UniqueID index = data.getUniqueID(i)
		UniqueID L = data.getUniqueID(0)
		GroundAtom truth = labelsDB.getAtom(label, index, L)
		GroundAtom pred = trainDB.getAtom(label, index, L)
		System.out.println("Error: " + Math.abs(pred.getValue() - truth.getValue()) + ", Pred: " + pred.getValue() + ", Truth: " + truth.getValue())
	}
}

trainDB.close()
labelsDB.close()

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
		//			HardEMRandOM2 hardRandOM = new HardEMRandOM2(m, db, labelsDB, config)
		//			hardRandOM.setSlackPenalty(10000)
		//			hardRandOM.learn()
			break
		case "FirstOrderRandOM":
			FirstOrderMetropolisRandOM randOM = new FirstOrderMetropolisRandOM(m, db, labelsDB, config)
			randOM.learn()
			break
		case "IncompatibilityRandOM":
		//			GroundIncompatibilityMetropolisRandOM randOM = new GroundIncompatibilityMetropolisRandOM(m, db, labelsDB, config);
			IncompatibilityMetropolisRandOM randOM = new IncompatibilityMetropolisRandOM(m, db, labelsDB, config);
			randOM.learn();
			break
		case "GroundRandOM":
			GroundMetropolisRandOM randOM = new GroundMetropolisRandOM(m, db, labelsDB, config)
			randOM.learn()
			break
		default:
			throw new IllegalArgumentException("Unrecognized method.");
	}
}
